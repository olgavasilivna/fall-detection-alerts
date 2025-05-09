import os
import cv2
import numpy as np
import mediapipe as mp
import time
import math
from typing import Tuple, List, Optional
from dotenv import load_dotenv
from alerts import AlertSystem
from fall_detector import FallDetector

# Load environment variables
load_dotenv()

class FallDetectionSystem:
    def __init__(self):
        """Initialize the fall detection system."""
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize alert system
        self.alert_system = AlertSystem()
        
        # Initialize fall detector with custom config
        self.fall_detector = FallDetector({
            'min_confidence': 0.5,
            'min_consecutive_frames': 1,  # Single frame detection
            'window_size': 5,  # Smaller window for faster response
            'base_velocity_threshold': 0.015,
            'base_torso_angle': 45,
            'debug': True
        })
        
        # Keypoint indices for pose detection
        self.LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP
        self.RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP
        self.LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER
        self.RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        self.LEFT_ANKLE = self.mp_pose.PoseLandmark.LEFT_ANKLE
        self.RIGHT_ANKLE = self.mp_pose.PoseLandmark.RIGHT_ANKLE
        self.NOSE = self.mp_pose.PoseLandmark.NOSE
        
        # State tracking
        self.previous_center_y = None
        self.current_bbox = None
        
    def calculate_bbox(self, landmarks):
        """Calculate bounding box from landmarks."""
        if not landmarks:
            return None
            
        # Get all y-coordinates
        y_coords = []
        x_coords = []
        
        # Add shoulder coordinates
        if landmarks.landmark[self.LEFT_SHOULDER].visibility > 0.5:
            y_coords.append(landmarks.landmark[self.LEFT_SHOULDER].y)
            x_coords.append(landmarks.landmark[self.LEFT_SHOULDER].x)
        if landmarks.landmark[self.RIGHT_SHOULDER].visibility > 0.5:
            y_coords.append(landmarks.landmark[self.RIGHT_SHOULDER].y)
            x_coords.append(landmarks.landmark[self.RIGHT_SHOULDER].x)
            
        # Add hip coordinates
        if landmarks.landmark[self.LEFT_HIP].visibility > 0.5:
            y_coords.append(landmarks.landmark[self.LEFT_HIP].y)
            x_coords.append(landmarks.landmark[self.LEFT_HIP].x)
        if landmarks.landmark[self.RIGHT_HIP].visibility > 0.5:
            y_coords.append(landmarks.landmark[self.RIGHT_HIP].y)
            x_coords.append(landmarks.landmark[self.RIGHT_HIP].x)
            
        # Add ankle coordinates if available
        if hasattr(self, 'LEFT_ANKLE') and landmarks.landmark[self.LEFT_ANKLE].visibility > 0.5:
            y_coords.append(landmarks.landmark[self.LEFT_ANKLE].y)
            x_coords.append(landmarks.landmark[self.LEFT_ANKLE].x)
        if hasattr(self, 'RIGHT_ANKLE') and landmarks.landmark[self.RIGHT_ANKLE].visibility > 0.5:
            y_coords.append(landmarks.landmark[self.RIGHT_ANKLE].y)
            x_coords.append(landmarks.landmark[self.RIGHT_ANKLE].x)
        
        # Check if we have enough points
        if len(y_coords) < 2 or len(x_coords) < 2:
            return None
        
        # Calculate bounding box
        min_y = min(y_coords)
        max_y = max(y_coords)
        min_x = min(x_coords)
        max_x = max(x_coords)
        
        # Calculate center and size
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        # Calculate ratio (height/width)
        ratio = height / width if width > 0 else 0
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'area': width * height,
            'ratio': ratio,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }
        
    def calculate_confidence(self, landmarks) -> float:
        """Calculate confidence score based on multiple factors."""
        # Get visibility scores for all key points
        visibilities = [
            landmarks.landmark[self.LEFT_SHOULDER].visibility,
            landmarks.landmark[self.RIGHT_SHOULDER].visibility,
            landmarks.landmark[self.LEFT_HIP].visibility,
            landmarks.landmark[self.RIGHT_HIP].visibility,
            landmarks.landmark[self.LEFT_ANKLE].visibility,
            landmarks.landmark[self.RIGHT_ANKLE].visibility
        ]
        
        # Calculate base confidence as minimum visibility
        base_confidence = min(visibilities)
        
        # Calculate position stability (how well the points form a coherent pose)
        shoulder_width = abs(landmarks.landmark[self.LEFT_SHOULDER].x - 
                           landmarks.landmark[self.RIGHT_SHOULDER].x)
        hip_width = abs(landmarks.landmark[self.LEFT_HIP].x - 
                       landmarks.landmark[self.RIGHT_HIP].x)
        
        # Check if proportions make sense (shoulders should be wider than hips)
        proportion_score = 1.0 if shoulder_width > hip_width else 0.5
        
        # Calculate final confidence
        confidence = base_confidence * proportion_score
        
        return confidence

    def calculate_torso_angle(self, landmarks):
        """Calculate the angle between shoulders and hips."""
        left_shoulder = landmarks.landmark[self.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.LEFT_HIP]
        right_hip = landmarks.landmark[self.RIGHT_HIP]

        shoulder_mid = [(left_shoulder.x + right_shoulder.x) / 2,
                       (left_shoulder.y + right_shoulder.y) / 2]
        hip_mid = [(left_hip.x + right_hip.x) / 2,
                  (left_hip.y + right_hip.y) / 2]

        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        angle = math.degrees(math.atan2(dy, dx))
        return abs(angle)

    def detect_fall(self, results) -> Tuple[bool, float]:
        """Analyze pose keypoints to detect a fall."""
        if not results.pose_landmarks:
            self.previous_center_y = None
            return False, 0.0
            
        # Get landmarks
        landmarks = results.pose_landmarks
        
        # Calculate confidence
        confidence = self.calculate_confidence(landmarks)
        
        # Calculate current bounding box
        current_bbox = self.calculate_bbox(landmarks)
        if current_bbox is None:
            return False, confidence
            
        # Calculate vertical velocity
        current_center_y = current_bbox['center_y']
        vertical_velocity = 0.0
        if self.previous_center_y is not None:
            vertical_velocity = current_center_y - self.previous_center_y
        self.previous_center_y = current_center_y
        
        # Get head and hip positions
        head_y = landmarks.landmark[self.NOSE].y
        hip_y = (landmarks.landmark[self.LEFT_HIP].y + landmarks.landmark[self.RIGHT_HIP].y) / 2
        
        # Calculate torso angle
        torso_angle = self.calculate_torso_angle(landmarks)
        
        # Prepare metrics for fall detector
        metrics = {
            'confidence': confidence,
            'vertical_velocity': vertical_velocity,
            'torso_angle': torso_angle,
            'head_y': head_y,
            'hip_y': hip_y,
            'current_ratio': current_bbox['ratio']
        }
        
        # Use fall detector to determine if fall occurred
        return self.fall_detector.is_fall_condition_met(metrics)

    def draw_keypoints(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw detected keypoints and bounding box on the frame."""
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Draw bounding box if available
            if self.current_bbox is not None:
                h, w = frame.shape[:2]
                # Convert normalized coordinates to pixel coordinates
                x1 = int(self.current_bbox['min_x'] * w)
                y1 = int(self.current_bbox['min_y'] * h)
                x2 = int(self.current_bbox['max_x'] * w)
                y2 = int(self.current_bbox['max_y'] * h)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw ratio text
                ratio_text = f"H/W Ratio: {self.current_bbox['ratio']:.2f}"
                cv2.putText(frame, ratio_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw height and width
                height_text = f"Height: {self.current_bbox['height']:.2f}"
                width_text = f"Width: {self.current_bbox['width']:.2f}"
                cv2.putText(frame, height_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, width_text, (x1, y2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """Process a single frame for fall detection."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Calculate current bounding box
        self.current_bbox = self.calculate_bbox(results.pose_landmarks) if results.pose_landmarks else None
        
        # Detect fall
        is_fall, confidence = self.detect_fall(results)
        
        # Calculate average score
        score_history = self.fall_detector.score_history
        avg_score = 0.0  # Initialize with default value
        if score_history:
            avg_score = sum(score_history) / len(score_history)
        
        # Draw keypoints first to create annotated frame
        annotated_frame = self.draw_keypoints(frame.copy(), results)
        
        # Send alert if new fall is detected
        if is_fall:
            print("\n" + "!"*50)
            print("!"*20 + " FALL DETECTED " + "!"*20)
            print("!"*50)
            print(f"Confidence: {confidence:.2f}")
            print(f"Score: {avg_score:.2f}")
            print("!"*50 + "\n")
            
            # Save frame for alert
            frame_path = f"fall_frame_{int(time.time())}.jpg"
            cv2.imwrite(frame_path, annotated_frame)
            
            try:
                # Send alert immediately
                self.alert_system.send_alert_sync(confidence, frame_path)
                print("!"*20 + " Alert sent successfully! " + "!"*20 + "\n")
            except Exception as e:
                print(f"Error sending alert: {e}")
            finally:
                # Clean up saved frame
                if os.path.exists(frame_path):
                    os.remove(frame_path)
        
        # Check for recovery alert
        if self.fall_detector.recovery_alert_sent:
            print("\n" + "="*50)
            print("="*20 + " RECOVERY DETECTED " + "="*20)
            print("="*50 + "\n")
            
            try:
                # Send recovery alert
                self.alert_system.send_recovery_alert({
                    'confidence': confidence,
                    'score': avg_score
                })
                print("="*20 + " Recovery alert sent successfully! " + "="*20 + "\n")
            except Exception as e:
                print(f"Error sending recovery alert: {e}")
            
            # Reset the recovery alert flag
            self.fall_detector.recovery_alert_sent = False
        
        return is_fall, confidence, annotated_frame

    def run(self, source: int = 0):
        """Run the fall detection system on video input."""
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame. Attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(source)
                continue
            
            # Process frame
            is_fall, confidence, annotated_frame = self.process_frame(frame)
            
            # Get current metrics for display
            current_bbox = None
            if hasattr(self, 'current_bbox'):
                current_bbox = self.current_bbox
            
            # Add debug information on frame
            bbox_info = ""
            if current_bbox is not None:
                bbox_info = (f"Ratio: {current_bbox['ratio']:.3f}")
                # Add warning when ratio deviates significantly from 1.0
                if abs(current_bbox['ratio'] - 1.0) > 0.3:  # More than 30% deviation
                    bbox_info += " (SIGNIFICANT DEVIATION!)"
            
            # Get fall score history for visualization
            score_history = self.fall_detector.score_history
            avg_score = 0.0  # Initialize with default value
            if score_history:
                # Calculate average score over the window
                avg_score = sum(score_history) / len(score_history)
            
            status_text = f"Fall: {is_fall}, Conf: {confidence:.2f}, Score: {avg_score:.2f}, {bbox_info}"
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not is_fall else (0, 0, 255), 2)
            
            # Add ratio explanation
            ratio_text = "Ratio: N/A"
            if self.current_bbox is not None:
                ratio_text = f"Ratio Target: 2.0 (Current: {self.current_bbox['ratio']:.1f})"
            cv2.putText(annotated_frame, ratio_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add confidence explanation
            confidence_explanation = f"Confidence > {self.fall_detector.config['min_confidence']} required"
            cv2.putText(annotated_frame, confidence_explanation, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Fall Detection', annotated_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FallDetectionSystem()
    detector.run() 