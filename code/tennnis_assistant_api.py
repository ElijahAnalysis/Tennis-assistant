import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import cv2
import shutil
import os
import pickle
import math
import pandas as pd

from ultralytics import YOLO

app = FastAPI()

class VideoReader:
    @staticmethod
    def read_video(video_capture):
        frames = []
        while True:
            read_successful, current_frame = video_capture.read()
            if not read_successful:
                break
            frames.append(current_frame)
        video_capture.release()
        return frames

    @staticmethod
    def read_video_from_path(video_path):
        """Read video frames directly from file path"""
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        while True:
            read_successful, current_frame = video_capture.read()
            if not read_successful:
                break
            frames.append(current_frame)
        video_capture.release()
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
        id_name_dict = results.names
        ball_dict = {}
        
        for box in results.boxes:
            class_id = box.cls.tolist()[0]
            class_name = id_name_dict[class_id]
            
            # Only detect if it's explicitly classified as 'ball'
            if class_name == 'ball':
                ball_dict[1] = box.xyxy.tolist()[0]
                # Only take the first ball detection to avoid duplicates
                break
                
        return ball_dict

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

class NetDetector:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)

    def detect_net(self, frames, read_from_stub=False, stub_path=None):
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
        net_dict = {}

        for box in results.boxes:
            net_dict[1] = box.xyxy.tolist()[0]
        return net_dict

class CourtLineDetector:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)

    def debug_classes(self, frame):
        """Debug function to see what classes are being detected"""
        results = self.model.predict(frame, conf=0.15)[0]
        id_name_dict = results.names
        detected_classes = {}
        
        for box in results.boxes:
            class_id = box.cls.tolist()[0]
            class_name = id_name_dict[class_id]
            if class_name in detected_classes:
                detected_classes[class_name] += 1
            else:
                detected_classes[class_name] = 1
        
        print(f"Detected classes: {detected_classes}")
        return detected_classes

    def detect_points(self, frames, read_from_stub=False, stub_path=None):
        detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                detections = pickle.load(f)
            return detections
        
        # Debug first frame to see what classes are detected
        if frames:
            print("Debugging first frame classes:")
            self.debug_classes(frames[0])
        
        for frame in frames:
            detections.append(self.detect_single_frame(frame))

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)

        return detections    

    def detect_single_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        id_name_dict = results.names
        point_dict = {}

        # Get all available classes for debugging
        available_classes = set(id_name_dict.values())
        print(f"Available classes in model: {available_classes}")
        
        # Exclude known non-keypoint classes
        non_keypoint_classes = {'player', 'ball'}
        
        # Simple counter for unique keypoint IDs
        keypoint_counter = 0
        
        for box in results.boxes:
            class_id = int(box.cls.tolist()[0])
            class_name = id_name_dict[class_id]

            # Include ALL detections that are NOT player, ball
            if class_name not in non_keypoint_classes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Just assign sequential IDs to all keypoints regardless of their class
                point_dict[keypoint_counter] = [cx, cy]
                
                confidence = box.conf.tolist()[0] if box.conf is not None else 0.0
                print(f"Keypoint {keypoint_counter}: original_class={class_name}, center=({cx:.1f}, {cy:.1f}), conf={confidence:.3f}")
                
                keypoint_counter += 1
            
        return point_dict

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
        """Calculate the pixel height of a player's bounding box"""
        x1, y1, x2, y2 = bbox
        return y2 - y1
    
    def get_court_pixel_length(self, court_points):
        """
        Calculate the pixel length of the court using detected points
        Uses the maximum distance between any two points as approximation
        Now handles multiple detections per class
        """
        if not court_points:
            return 500  # Default fallback value
            
        points = list(court_points.values())
        if len(points) < 2:
            return 500  # Default fallback value
            
        max_distance = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                max_distance = max(max_distance, distance)
        
        return max_distance if max_distance > 0 else 500
    
    def get_perspective_scale_factor(self, bbox, track_id):
        """
        Calculate the perspective scale factor (meters per pixel) based on player height
        """
        height_pixels = self.get_player_height_in_pixels(bbox)
        real_height = self.player_heights.get(track_id, self.average_player_height)
        meters_per_pixel = real_height / height_pixels
        return meters_per_pixel
    
    def pixels_to_meters_height_reference(self, pixel_distance, meters_per_pixel):
        """Convert pixel distance to real-world meters using height reference"""
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
    
    def estimate_player_speeds(self, player_detections, court_points):
        """
        Estimate speeds for all players across all frames
        Uses player height for better accuracy when available
        """
        court_pixel_length = self.get_court_pixel_length(court_points)
        player_speeds = []
        
        for frame_idx in range(len(player_detections)):
            frame_speeds = {}
            
            if frame_idx == 0:
                for track_id in player_detections[frame_idx]:
                    frame_speeds[track_id] = 0.0
            else:
                current_frame = player_detections[frame_idx]
                previous_frame = player_detections[frame_idx - 1]
                
                for track_id in current_frame:
                    if track_id in previous_frame:
                        current_center = self.calculate_center(current_frame[track_id])
                        previous_center = self.calculate_center(previous_frame[track_id])
                        
                        pixel_distance = self.calculate_distance(current_center, previous_center)
                        
                        meters_per_pixel = self.get_perspective_scale_factor(
                            current_frame[track_id], track_id
                        )
                        real_distance = self.pixels_to_meters_height_reference(
                            pixel_distance, meters_per_pixel
                        )
                        
                        speed_ms = real_distance * self.fps
                        speed_kmh = speed_ms * 3.6
                        
                        frame_speeds[track_id] = speed_kmh
                    else:
                        frame_speeds[track_id] = 0.0
            
            player_speeds.append(frame_speeds)
        
        return player_speeds
    
    def estimate_ball_speeds(self, ball_detections, court_points):
        """
        Estimate ball speeds across all frames
        Uses court reference for ball since height reference doesn't apply
        """
        court_pixel_length = self.get_court_pixel_length(court_points)
        ball_speeds = []
        
        for frame_idx in range(len(ball_detections)):
            if frame_idx == 0 or 1 not in ball_detections[frame_idx]:
                ball_speeds.append(0.0)
            else:
                current_frame = ball_detections[frame_idx]
                previous_frame = ball_detections[frame_idx - 1]
                
                if 1 in previous_frame and 1 in current_frame:
                    current_center = self.calculate_center(current_frame[1])
                    previous_center = self.calculate_center(previous_frame[1])
                    
                    pixel_distance = self.calculate_distance(current_center, previous_center)
                    
                    real_distance = self.pixels_to_meters_court_reference(
                        pixel_distance, court_pixel_length
                    )
                    
                    speed_ms = real_distance * self.fps
                    speed_kmh = speed_ms * 3.6
                    
                    ball_speeds.append(speed_kmh)
                else:
                    ball_speeds.append(0.0)
        
        return ball_speeds

class Visualizer:
    def __init__(self, ball_color=(0, 255, 255), thickness=3):
        self.ball_color = ball_color
        self.thickness = thickness
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
            
            label = 'Ball'
            if ball_speed is not None:
                label += f' ({ball_speed:.1f} km/h)'
            
            cv2.putText(frame, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, self.ball_color, 2, cv2.LINE_AA)
        return frame
    
    def draw_court_points(self, frame, court_points):
        """Draw detected court line points - handles multiple detections per class"""
        for point_key, (x, y) in court_points.items():
            cv2.circle(frame, (int(x), int(y)), 8, (0, 140, 255), -1)
            # Display the point key (could be class_id or class_id_N for multiple detections)
            cv2.putText(frame, str(point_key), (int(x) + 10, int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        return frame

    def draw_net(self, frame, net_dict):
        """Draw tennis net"""
        if 1 in net_dict:
            x1, y1, x2, y2 = map(int, net_dict[1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), self.thickness)
            cv2.putText(frame, 'Net', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def draw_all(self, frame, player_dict, ball_dict, court_points=None, net_dict=None, 
                player_speeds=None, ball_speed=None):
        frame = self.draw_players(frame, player_dict, player_speeds)
        frame = self.draw_ball(frame, ball_dict, ball_speed)
        if court_points is not None:
            frame = self.draw_court_points(frame, court_points)
        if net_dict is not None:
            frame = self.draw_net(frame, net_dict)
        return frame

def save_video(output_path, frames, player_detections, ball_detections, visualizer, 
            fps=30, court_detections=None, net_detections=None, 
            player_speeds=None, ball_speeds=None):
    """
    Save annotated video with all detections and speeds
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, (frame, player_dict, ball_dict) in enumerate(zip(frames, player_detections, ball_detections)):
        frame_copy = frame.copy()
        
        # Get current frame data
        current_player_speeds = player_speeds[i] if player_speeds else None
        current_ball_speed = ball_speeds[i] if ball_speeds else None
        current_court_points = court_detections[i] if court_detections else None
        current_net_dict = net_detections[i] if net_detections else None
        
        # Draw all annotations
        drawn_frame = visualizer.draw_all(
            frame_copy, 
            player_dict, 
            ball_dict, 
            current_court_points, 
            current_net_dict,
            current_player_speeds, 
            current_ball_speed
        )

        out.write(drawn_frame)

    out.release()
    print(f"Video saved to: {output_path}")

# Initialize detectors globally for API
yolo11n_bpk_path = r"C:\Users\User\Desktop\Github\Tennis-assistant\models\yolo11n_bpk.pt"
yolo11n_net_path = r"C:\Users\User\Desktop\Github\Tennis-assistant\models\yolo11n_net_detector\weights\best.pt"

player_detector = PlayerDetector(yolo11n_bpk_path)
ball_detector = BallDetector(yolo11n_bpk_path)
court_line_detector = CourtLineDetector(yolo11n_bpk_path)
net_detector = NetDetector(yolo11n_net_path)

class PlayerHeights(BaseModel):
    first_player_height_in_meters: float
    second_player_height_in_meters: float
    third_player_height_in_meters: float | None = None
    fourth_player_height_in_meters: float | None = None

@app.get('/get_metadata/{filename}')
async def get_metadata(filename: str):
    """Get metadata for a processed video"""
    metadata_path = rf"C:\Users\User\Downloads\temp_videos\{filename.replace('.mp4', '_metadata.json')}"
    
    if not os.path.exists(metadata_path):
        return JSONResponse(
            status_code=404,
            content={"error": "Metadata not found for this video"}
        )
    
    with open(metadata_path, 'r') as f:
        import json
        metadata = json.load(f)
    
    return JSONResponse(content=metadata)

@app.post('/annotate_video_with_zip')
async def annotate_video_with_zip(
    video_file: UploadFile = File(...),
    first_player_height_in_meters: float = Form(...),
    second_player_height_in_meters: float = Form(...),
    third_player_height_in_meters: float = Form(None),
    fourth_player_height_in_meters: float = Form(None)
):
    """Alternative endpoint that returns a ZIP file containing both video and metadata"""
    import zipfile
    
    # Save uploaded file
    save_path = rf"C:\Users\User\Downloads\temp_videos\{video_file.filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    # Read video
    video_capture = cv2.VideoCapture(save_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = VideoReader.read_video(video_capture)

    # Create heights dictionary from form data
    heights_dict = {
        1: first_player_height_in_meters,
        2: second_player_height_in_meters,
    }
    if third_player_height_in_meters and third_player_height_in_meters > 0:
        heights_dict[3] = third_player_height_in_meters
    if fourth_player_height_in_meters and fourth_player_height_in_meters > 0:
        heights_dict[4] = fourth_player_height_in_meters

    # --- Detection ---
    player_detections = player_detector.detect_players(frames)
    ball_detections = ball_detector.detect_ball(frames)
    ball_detections = ball_detector.interpolate_ball_positions(ball_detections)
    net_detections = net_detector.detect_net(frames)
    court_detections = court_line_detector.detect_points(frames)
    first_frame_court_points = court_detections[0] if court_detections else {}

    # --- Speed Estimation ---
    speed_estimator = SpeedEstimator(
        fps=fps, 
        player_heights=heights_dict,
        average_player_height=1.75
    )
    player_speeds = speed_estimator.estimate_player_speeds(player_detections, first_frame_court_points)
    ball_speeds = speed_estimator.estimate_ball_speeds(ball_detections, first_frame_court_points)

    # --- Visualization ---
    visualizer = Visualizer()
        
    # --- Save annotated video ---
    output_path = save_path.replace(".mp4", "_annotated.mp4")
    save_video(
        output_path, 
        frames, 
        player_detections, 
        ball_detections, 
        visualizer,
        fps=fps,
        court_detections=court_detections,
        net_detections=net_detections,
        player_speeds=player_speeds,
        ball_speeds=ball_speeds
    )

    # Save metadata
    metadata_path = save_path.replace(".mp4", "_metadata.json")
    metadata = {
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": len(frames)
        },
        "player_heights": heights_dict,
        "court_keypoint_detections": court_detections,
        "player_detections": player_detections,
        "ball_detections": ball_detections,
        "net_detections": net_detections,
        "player_speeds": player_speeds,
        "ball_speeds": ball_speeds
    }
    
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)

    # Create ZIP file
    zip_path = save_path.replace(".mp4", "_result.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(output_path, os.path.basename(output_path))
        zipf.write(metadata_path, os.path.basename(metadata_path))

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path)
    )

@app.post('/annotate_video')
async def annotate_video(
    video_file: UploadFile = File(...),
    first_player_height_in_meters: float = Form(...),
    second_player_height_in_meters: float = Form(...),
    third_player_height_in_meters: float = Form(None),
    fourth_player_height_in_meters: float = Form(None)
):
    # Save uploaded file
    save_path = rf"C:\Users\User\Downloads\temp_videos\{video_file.filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    # Read video
    video_capture = cv2.VideoCapture(save_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = VideoReader.read_video(video_capture)

    # Create heights dictionary from form data
    heights_dict = {
        1: first_player_height_in_meters,
        2: second_player_height_in_meters,
    }
    if third_player_height_in_meters and third_player_height_in_meters > 0:
        heights_dict[3] = third_player_height_in_meters
    if fourth_player_height_in_meters and fourth_player_height_in_meters > 0:
        heights_dict[4] = fourth_player_height_in_meters

    # --- Detection ---
    player_detections = player_detector.detect_players(frames)
    ball_detections = ball_detector.detect_ball(frames)
    ball_detections = ball_detector.interpolate_ball_positions(ball_detections)
    net_detections = net_detector.detect_net(frames)
    court_detections = court_line_detector.detect_points(frames)
    first_frame_court_points = court_detections[0] if court_detections else {}

    # --- Speed Estimation ---
    speed_estimator = SpeedEstimator(
        fps=fps, 
        player_heights=heights_dict,
        average_player_height=1.75  # Default height for unknown players
    )
    player_speeds = speed_estimator.estimate_player_speeds(player_detections, first_frame_court_points)
    ball_speeds = speed_estimator.estimate_ball_speeds(ball_detections, first_frame_court_points)

    # --- Visualization ---
    visualizer = Visualizer()
    
    # --- Save annotated video ---
    output_path = save_path.replace(".mp4", "_annotated.mp4")
    save_video(
        output_path, 
        frames, 
        player_detections, 
        ball_detections, 
        visualizer,
        fps=fps,
        court_detections=court_detections,
        net_detections=net_detections,
        player_speeds=player_speeds,
        ball_speeds=ball_speeds
    )

    # --- Return final video file ---
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=os.path.basename(output_path)
    )



