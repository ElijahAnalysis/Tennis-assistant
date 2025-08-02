import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch 
from ultralytics import YOLO
import cv2
import torchvision

import numpy as np
import pandas as pd 
import scipy as scp
import pickle
import math

class VideoReader:

    @staticmethod
    def read_video(video_path):
        """
        Reads a video file from the given path and returns a list of frames.
        
        Parameters:
        - video_path (str): Path to the video file.
        
        Returns:
        - frames (list): A list containing all frames from the video.
        """
        video_capture = cv2.VideoCapture(video_path)  # Open video file using OpenCV
        frames = []  # List to store all the frames
        
        while True:
            read_successful, current_frame = video_capture.read()  # Read each frame
            if not read_successful:
                break  # Stop when no frames are left
            frames.append(current_frame)  # Add the current frame to the list
        
        video_capture.release()  # Release the video file from memory
        return frames


class PlayerDetector:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)

    def detect_players(self, frames, read_from_stub=False, stub_path=None):
        detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                detections = pickle.load(f)
            return detections

        for frame in frames:
            detections.append(self.detect_single_frame(frame))

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)

        return detections

    def detect_single_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            cls_id = box.cls.tolist()[0]
            cls_name = id_name_dict[cls_id]
            if cls_name == "player":
                player_dict[track_id] = box.xyxy.tolist()[0]

        return player_dict


class BallDetector:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)

    def detect_ball(self, frames, read_from_stub=False, stub_path=None):
        detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                detections = pickle.load(f)
            return detections

        for frame in frames:
            detections.append(self.detect_single_frame(frame))

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)

        return detections

    def detect_single_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            ball_dict[1] = box.xyxy.tolist()[0]
        return ball_dict

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        # Extracting ball position
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            neg_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            pos_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if neg_change or pos_change:
                change_count = 0
                for j in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    neg_follow = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[j] < 0
                    pos_follow = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[j] > 0
                    if neg_change and neg_follow:
                        change_count += 1
                    elif pos_change and pos_follow:
                        change_count += 1
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.at[i, 'ball_hit'] = 1

        return df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = torchvision.models.efficientnet_b4(pretrained=True)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

        h, w = image.shape[:2]
        image_tensor = self.transform(image).unsqueeze(0)  # Keep BGR image here
        with torch.no_grad():
            outputs = self.model(image_tensor).cpu().numpy().reshape(-1, 2)
        outputs[:, 0] *= w / 224.0
        outputs[:, 1] *= h / 224.0
        
        return outputs.flatten()


class SpeedEstimator:
    def __init__(self, fps=30, court_length_meters=23.77, player_heights=None, average_player_height=1.75):
        """
        Initialize speed estimator
        
        Parameters:
        - fps: frames per second of the video
        - court_length_meters: real-world length of tennis court in meters (baseline to baseline)
        - player_heights: dict mapping player track_id to height in meters (e.g., {1: 1.85, 2: 1.70})
        - average_player_height: default height in meters if specific height not known
        """
        self.fps = fps
        self.court_length_meters = court_length_meters
        self.player_heights = player_heights or {}
        self.average_player_height = average_player_height
        
    def get_player_height_in_pixels(self, bbox):
        """
        Calculate the pixel height of a player's bounding box
        """
        x1, y1, x2, y2 = bbox
        return y2 - y1
    
    def get_court_pixel_length(self, court_keypoints):
        """
        Calculate the pixel length of the court using keypoints
        Assumes keypoints are ordered and we can identify baseline points
        """
        # Reshape keypoints to (14, 2) format
        points = court_keypoints.reshape(-1, 2)
        
        # Calculate distance between opposite baselines (approximate)
        # You may need to adjust these indices based on your keypoint ordering
        baseline1 = points[0]  # top left baseline
        baseline2 = points[3]  # bottom right baseline
        
        pixel_distance = math.sqrt((baseline1[0] - baseline2[0])**2 + (baseline1[1] - baseline2[1])**2)
        return pixel_distance
    
    def get_perspective_scale_factor(self, bbox, track_id):
        """
        Calculate the perspective scale factor (meters per pixel) based on player height
        
        Parameters:
        - bbox: player bounding box [x1, y1, x2, y2]
        - track_id: player track ID
        
        Returns:
        - meters_per_pixel: scale factor for converting pixels to meters
        """
        # Get player height in pixels
        height_pixels = self.get_player_height_in_pixels(bbox)
        
        # Get real-world height for this player (use provided height or default to average)
        real_height = self.player_heights.get(track_id, self.average_player_height)
        
        # Calculate meters per pixel based on height
        meters_per_pixel = real_height / height_pixels
        
        return meters_per_pixel
    
    def pixels_to_meters_height_reference(self, pixel_distance, meters_per_pixel):
        """
        Convert pixel distance to real-world meters using height reference
        
        Parameters:
        - pixel_distance: distance in pixels
        - meters_per_pixel: scale factor from get_perspective_scale_factor
        
        Returns:
        - real_distance: distance in meters
        """
        return pixel_distance * meters_per_pixel
    
    def pixels_to_meters_court_reference(self, pixel_distance, court_pixel_length):
        """Convert pixel distance to real-world meters using court reference"""
        return (pixel_distance * self.court_length_meters) / court_pixel_length
    
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def estimate_player_speeds(self, player_detections, court_keypoints):
        """
        Estimate speeds for all players across all frames
        Uses player height for better accuracy when available
        
        Returns:
        - Dictionary with player speeds for each frame
        """
        court_pixel_length = self.get_court_pixel_length(court_keypoints)
        player_speeds = []
        
        for frame_idx in range(len(player_detections)):
            frame_speeds = {}
            
            if frame_idx == 0:
                # First frame, no previous position to compare
                for track_id in player_detections[frame_idx]:
                    frame_speeds[track_id] = 0.0
            else:
                current_frame = player_detections[frame_idx]
                previous_frame = player_detections[frame_idx - 1]
                
                for track_id in current_frame:
                    if track_id in previous_frame:
                        # Calculate centers
                        current_center = self.calculate_center(current_frame[track_id])
                        previous_center = self.calculate_center(previous_frame[track_id])
                        
                        # Calculate pixel distance
                        pixel_distance = self.calculate_distance(current_center, previous_center)
                        
                        # Use player height for conversion (always use height reference now)
                        meters_per_pixel = self.get_perspective_scale_factor(
                            current_frame[track_id], track_id
                        )
                        real_distance = self.pixels_to_meters_height_reference(
                            pixel_distance, meters_per_pixel
                        )
                        
                        # Calculate speed (distance per frame * fps = distance per second)
                        speed_ms = real_distance * self.fps  # meters per second
                        speed_kmh = speed_ms * 3.6  # kilometers per hour
                        
                        frame_speeds[track_id] = speed_kmh
                    else:
                        frame_speeds[track_id] = 0.0
            
            player_speeds.append(frame_speeds)
        
        return player_speeds
    
    def estimate_ball_speeds(self, ball_detections, court_keypoints):
        """
        Estimate ball speeds across all frames
        Uses court reference for ball since height reference doesn't apply
        
        Returns:
        - List of ball speeds for each frame
        """
        court_pixel_length = self.get_court_pixel_length(court_keypoints)
        ball_speeds = []
        
        for frame_idx in range(len(ball_detections)):
            if frame_idx == 0 or 1 not in ball_detections[frame_idx]:
                ball_speeds.append(0.0)
            else:
                current_frame = ball_detections[frame_idx]
                previous_frame = ball_detections[frame_idx - 1]
                
                if 1 in previous_frame and 1 in current_frame:
                    # Calculate centers
                    current_center = self.calculate_center(current_frame[1])
                    previous_center = self.calculate_center(previous_frame[1])
                    
                    # Calculate pixel distance
                    pixel_distance = self.calculate_distance(current_center, previous_center)
                    
                    # Convert to real-world distance using court reference
                    real_distance = self.pixels_to_meters_court_reference(
                        pixel_distance, court_pixel_length
                    )
                    
                    # Calculate speed
                    speed_ms = real_distance * self.fps  # meters per second
                    speed_kmh = speed_ms * 3.6  # kilometers per hour
                    
                    ball_speeds.append(speed_kmh)
                else:
                    ball_speeds.append(0.0)
        
        return ball_speeds


class Visualizer:
    def __init__(self, ball_color=(0, 255, 255), thickness=3):  # Bright yellow ball
        self.ball_color = ball_color
        self.thickness = thickness
        # Different colors for different players
        self.player_colors = [
            (255, 100, 100),   # Light Blue
            (100, 255, 100),   # Light Green
            (100, 100, 255),   # Light Red
            (255, 255, 100),   # Light Cyan
            (255, 100, 255),   # Light Magenta
            (100, 255, 255),   # Light Yellow
            (200, 150, 100),   # Light Brown
            (150, 100, 200),   # Light Purple
        ]

    def get_player_color(self, track_id):
        """Get color for a specific player based on track_id"""
        return self.player_colors[track_id % len(self.player_colors)]

    def draw_players(self, frame, player_dict, player_speeds=None):
        for track_id, coords in player_dict.items():
            x1, y1, x2, y2 = map(int, coords)
            color = self.get_player_color(track_id)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Display player ID and speed with bigger font
            label = f'Player {track_id}'
            if player_speeds and track_id in player_speeds:
                speed = player_speeds[track_id]
                label += f' ({speed:.1f} km/h)'
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)
        return frame

    def draw_ball(self, frame, ball_dict, ball_speed=None):
        if 1 in ball_dict:
            x1, y1, x2, y2 = map(int, ball_dict[1])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 8, self.ball_color, -1)
            
            # Display ball speed with bigger font
            label = 'Ball'
            if ball_speed is not None:
                label += f' ({ball_speed:.1f} km/h)'
            
            cv2.putText(frame, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, self.ball_color, 2, cv2.LINE_AA)
        return frame
    
    def draw_court_lines(self, frame, court_points):
        for x, y in court_points:
            cv2.circle(frame, (x, y), 8, (0, 140, 255), -1)  
        return frame

    def draw_all(self, frame, player_dict, ball_dict, court_points=None, player_speeds=None, ball_speed=None):
        frame = self.draw_players(frame, player_dict, player_speeds)
        frame = self.draw_ball(frame, ball_dict, ball_speed)
        if court_points is not None:
            frame = self.draw_court_lines(frame, court_points)
        return frame


def save_video(output_path, frames, player_detections, ball_detections, visualizer, court_keypoints, fps=30, court_points=None, player_speeds=None, ball_speeds=None):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, (frame, player_dict, ball_dict) in enumerate(zip(frames, player_detections, ball_detections)):
        frame_copy = frame.copy()
        
        # Get speeds for current frame
        current_player_speeds = player_speeds[i] if player_speeds else None
        current_ball_speed = ball_speeds[i] if ball_speeds else None
        
        # Draw annotations with speeds
        drawn_frame = visualizer.draw_all(
            frame_copy, 
            player_dict, 
            ball_dict, 
            court_points, 
            current_player_speeds, 
            current_ball_speed
        )

        out.write(drawn_frame)

    out.release()
    print(f"Video saved to: {output_path}")


# Main execution
video_path = r"C:\Users\User\Desktop\tennis assistant\data\tennis_game_sample2.mp4"
output_path = r"C:\Users\User\Downloads\annotated_output_with_speed.mp4"

player_detector = PlayerDetector(r"C:\Users\User\Desktop\tennis assistant\models\yolo11s_tennisv2.pt")
ball_detector = BallDetector(r"C:\Users\User\Desktop\tennis assistant\models\yolo11s_tennisv2.pt")

court_model_path = r"C:\Users\User\Desktop\tennis assistant\models\efficientnetb4_tennis_court_keypoints.pt"
court_line_detector = CourtLineDetector(court_model_path)

# Read video and get FPS
video_capture = cv2.VideoCapture(video_path)
fps = video_capture.get(cv2.CAP_PROP_FPS)
video_capture.release()

frames = VideoReader.read_video(video_path)

player_detections = player_detector.detect_players(frames)
ball_detections = ball_detector.detect_ball(frames)
ball_detections = ball_detector.interpolate_ball_positions(ball_detections)

# Detect court lines
court_keypoints = court_line_detector.predict(frames[0])
court_points = court_keypoints.reshape(-1, 2).astype(int)

# Initialize speed estimator with player heights (optional)
# Example: player heights in meters - adjust track_ids and heights as needed
player_heights = {
    1: 1.88,  
    2: 1.83,  
    # Add more players as needed
    # If not specified, will use average_player_height of 1.75
}

speed_estimator = SpeedEstimator(
    fps=fps, 
    player_heights=player_heights,
    average_player_height=1.75  # Default height for unknown players
)

# Calculate speeds
player_speeds = speed_estimator.estimate_player_speeds(player_detections, court_keypoints)
ball_speeds = speed_estimator.estimate_ball_speeds(ball_detections, court_keypoints)

# Save annotated video with speeds and court lines (no mini court)
visualizer = Visualizer()
save_video(
    output_path, 
    frames, 
    player_detections, 
    ball_detections, 
    visualizer,
    court_keypoints,
    fps=fps,
    court_points=court_points,
    player_speeds=player_speeds,
    ball_speeds=ball_speeds
)





