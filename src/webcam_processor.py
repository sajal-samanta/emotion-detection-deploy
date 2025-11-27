import cv2
import numpy as np
import threading
import time
from collections import deque

class WebcamProcessor:
    def __init__(self, emotion_detector):
        self.emotion_detector = emotion_detector
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        self.emotion_history = deque(maxlen=30)  # Store last 30 emotions
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_camera(self, camera_index=0):
        """Start webcam capture"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        self.is_running = True
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.start()
        
    def stop_camera(self):
        """Stop webcam capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def _process_frames(self):
        """Process frames in separate thread"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame
            processed_frame = self._detect_emotions_in_frame(frame)
            self.processed_frame = processed_frame
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
                
    def _detect_emotions_in_frame(self, frame):
        """Detect emotions in the current frame"""
        frame_copy = frame.copy()
        
        # Detect faces
        faces = self.emotion_detector.detect_faces(frame_copy)
        
        # Process each face
        for (x, y, w, h) in faces:
            try:
                # Extract face ROI
                face_roi = frame_copy[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, confidence, all_predictions = self.emotion_detector.detect_emotion(face_roi)
                
                # Store emotion in history
                self.emotion_history.append(emotion)
                
                # Draw emotion information
                self.emotion_detector.draw_emotion_info(
                    frame_copy, x, y, w, h, emotion, confidence, all_predictions
                )
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Add FPS counter
        cv2.putText(frame_copy, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add emotion statistics
        if self.emotion_history:
            dominant_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
            emotion_count = len([e for e in self.emotion_history if e == dominant_emotion])
            total_frames = len(self.emotion_history)
            percentage = (emotion_count / total_frames) * 100
            
            stats_text = f"Dominant: {dominant_emotion} ({percentage:.1f}%)"
            cv2.putText(frame_copy, stats_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_copy
    
    def get_processed_frame(self):
        """Get the latest processed frame"""
        return self.processed_frame
    
    def get_emotion_stats(self):
        """Get emotion statistics"""
        if not self.emotion_history:
            return {}
        
        stats = {}
        total = len(self.emotion_history)
        for emotion in set(self.emotion_history):
            count = self.emotion_history.count(emotion)
            stats[emotion] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        return stats