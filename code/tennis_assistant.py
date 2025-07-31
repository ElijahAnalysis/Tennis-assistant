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


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = torchvision.models.efficientnet_b4(pretrained=True)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):

    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0 
        keypoints[1::2] *= original_h / 224.0 

        return keypoints 



class MiniCourt:
    def __init__(self,frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_lines()
        self.drawing_key_points = self.generate_scaled_keypoints()


    def set_canvas_background_box_position(self,frame):
        frame= frame.copy() # not overriding the original frame

        self.end_x = frame.shape[1] - self.buffer 
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height    

    def set_mini_court_position(self):
        # Adding the padding to place the court corner points
  
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
        return (meters * refrence_height_in_pixels) / refrence_height_in_meters
    
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)]
        
    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle with the given coordinates
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5 #transparency level
        mask = shapes.astype(bool) # creates a boolean mask where the white rectangle is True and everything else is False.
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_court(self,frame):
            for i in range(0, len(self.drawing_key_points),2):
                x = int(self.drawing_key_points[i])
                y = int(self.drawing_key_points[i+1])
                cv2.circle(frame, (x,y),5, (0,0,255),-1) # draws point at respective position

            # draw Lines
            for line in self.lines:
                start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
                end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
                cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

            # Draw net
            net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
            net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
            cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

            return frame

    def generate_scaled_keypoints(self):
        """
        Return scaled dummy keypoints just for visualization.
        Replace with actual court keypoints if needed.
        """
        # Normalize court coordinates (assumes 14 points)
        # Replace with real keypoints if desired
        normalized = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],  # corners
            [0.1, 0.5], [0.9, 0.5], [0.5, 0.1], [0.5, 0.9],  # midpoints
            [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],  # inside box
            [0.5, 0.5], [0.5, 0.8]                           # net
        ])  # shape: (14, 2)

        # Scale to fit inside the mini court
        scaled = []
        for x, y in normalized:
            px = int(self.court_start_x + x * self.court_drawing_width)
            py = int(self.court_start_y + y * (self.court_end_y - self.court_start_y))
            scaled.extend([px, py])
        return scaled



class Visualizer:
    def __init__(self, player_color=(0, 255, 0), ball_color=(0, 0, 255), thickness=2):
        self.player_color = player_color
        self.ball_color = ball_color
        self.thickness = thickness

    def draw_players(self, frame, player_dict):
        for track_id, coords in player_dict.items():
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.player_color, self.thickness)
            cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self.player_color, 1, cv2.LINE_AA)
        return frame

    def draw_ball(self, frame, ball_dict):
        if 1 in ball_dict:
            x1, y1, x2, y2 = map(int, ball_dict[1])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, self.ball_color, -1)
            cv2.putText(frame, 'Ball', (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self.ball_color, 1, cv2.LINE_AA)
        return frame
    
    def draw_court_lines(self, frame, court_points):
        for x, y in court_points:
            cv2.circle(frame, (x, y), 4, (0, 140, 255), -1)  # light blue dots
        return frame

    def draw_all(self, frame, player_dict, ball_dict, court_points=None):
        frame = self.draw_players(frame, player_dict)
        frame = self.draw_ball(frame, ball_dict)
        if court_points is not None:
            frame = self.draw_court_lines(frame, court_points)
        return frame
    


def save_video(output_path, frames, player_detections, ball_detections, visualizer, fps=30, court_points=None):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame, player_dict, ball_dict in zip(frames, player_detections, ball_detections):
        frame_copy = frame.copy()
        drawn_frame = visualizer.draw_all(frame_copy, player_dict, ball_dict, court_points)

        # Draw mini-court on top of frame
        minicourt = MiniCourt(drawn_frame)
        minicourt.drawing_key_points = court_keypoints.astype(int).flatten().tolist()
        drawn_frame = minicourt.draw_mini_court([drawn_frame])[0]

        out.write(drawn_frame)

    out.release()
    print(f"Video saved to: {output_path}")


video_path = r"C:\Users\User\Desktop\tennis assistant\data\tennis_game_sample2.mp4"
output_path = r"C:\Users\User\Downloads\annotated_output.mp4"


player_detector = PlayerDetector(r"C:\Users\User\Desktop\tennis assistant\models\yolo11s_tennisv2.pt")
ball_detector = BallDetector(r"C:\Users\User\Desktop\tennis assistant\models\yolo11s_tennisv2.pt")

court_model_path = r"C:\Users\User\Desktop\tennis assistant\models\efficientnetb4_tennis_court_keypoints.pt"
court_line_detector = CourtLineDetector(court_model_path)

frames = VideoReader.read_video(video_path)

player_detections = player_detector.detect_players(frames)
ball_detections = ball_detector.detect_ball(frames)
ball_detections = ball_detector.interpolate_ball_positions(ball_detections)

# Detect court lines
court_keypoints = court_line_detector.predict(frames[0])
court_points = court_keypoints.reshape(-1, 2).astype(int)

# Save annotated video
visualizer = Visualizer()
save_video(output_path, frames, player_detections, ball_detections, visualizer, court_points=court_points)














