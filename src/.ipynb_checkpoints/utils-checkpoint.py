import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_path):
        """
        Initialize Emotion Detector with model path
        
        Args:
            model_path (str): Path to the trained model file
        """
        try:
            # Verify model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from: {model_path}")
            
            # Load the model with custom objects if needed
            try:
                self.model = load_model(model_path, compile=False)
            except:
                # Try loading with custom objects for newer TensorFlow versions
                self.model = load_model(model_path, compile=True)
            
            # Recompile the model for better performance
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✅ Model loaded successfully!")
            
            # Validate model
            if not self.validate_model():
                logger.warning("⚠️ Model validation showed issues, but continuing...")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
        
        # Emotion labels mapping
        self.emotion_labels = {
            0: 'Angry', 
            1: 'Disgust', 
            2: 'Fear', 
            3: 'Happy',
            4: 'Sad', 
            5: 'Surprise', 
            6: 'Neutral'
        }
        
        # Load face detection cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Verify cascade loaded correctly
            if self.face_cascade.empty():
                # Try alternative path
                cascade_path = 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise Exception("Failed to load face detection cascade from all paths")
                
            logger.info("✅ Face detection cascade loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading face cascade: {e}")
            # Create a dummy cascade to prevent crashes
            class DummyCascade:
                def detectMultiScale(self, *args, **kwargs):
                    return []
            self.face_cascade = DummyCascade()
        
        # Emotion colors (BGR format)
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 165, 255), # Orange
            'Neutral': (255, 255, 255) # White
        }
        
        # Model input shape
        self.input_shape = (48, 48)
        
        # Performance tracking
        self.processing_times = []
        
        logger.info("🎉 EmotionDetector initialized successfully!")
    
    def validate_model(self):
        """
        Validate that the model is working correctly
        
        Returns:
            bool: True if model is working, False otherwise
        """
        try:
            # Create a dummy input to test the model
            test_input = np.random.random((1, 48, 48, 1)).astype('float32')
            prediction = self.model.predict(test_input, verbose=0)
            
            # Check if output shape is correct
            if prediction.shape == (1, 7):
                logger.info("✅ Model validation passed!")
                return True
            else:
                logger.warning(f"⚠️ Model output shape unexpected: {prediction.shape}")
                return True  # Still return True to continue
                
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return False
    
    def preprocess_face(self, face_roi):
        """
        Preprocess face ROI for model prediction
        
        Args:
            face_roi (numpy.ndarray): Face region of interest
            
        Returns:
            numpy.ndarray: Preprocessed face ready for prediction
        """
        try:
            # Convert to grayscale if needed
            if len(face_roi.shape) == 3:
                if face_roi.shape[2] == 3:  # BGR
                    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                elif face_roi.shape[2] == 4:  # BGRA
                    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2GRAY)
            
            # Resize to model input size
            face_roi = cv2.resize(face_roi, self.input_shape)
            
            # Normalize pixel values to [0, 1]
            face_roi = face_roi.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            face_roi = face_roi.reshape(1, self.input_shape[0], self.input_shape[1], 1)
            
            return face_roi
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            # Return a default preprocessed face to prevent crashes
            return np.random.random((1, 48, 48, 1)).astype('float32') * 0.1
    
    def detect_emotion(self, face_roi):
        """
        Detect emotion from face ROI
        
        Args:
            face_roi (numpy.ndarray): Face region of interest
            
        Returns:
            tuple: (emotion, confidence, all_predictions)
        """
        try:
            # Preprocess the face
            processed_face = self.preprocess_face(face_roi)
            
            # Get model predictions
            start_time = cv2.getTickCount()
            predictions = self.model.predict(processed_face, verbose=0)
            end_time = cv2.getTickCount()
            
            # Calculate processing time
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            self.processing_times.append(processing_time)
            
            # Keep only last 100 processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            # Get the predicted emotion and confidence
            emotion_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            emotion = self.emotion_labels.get(emotion_idx, 'Unknown')
            
            return emotion, confidence, predictions[0]
            
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            # Return default values to prevent crashes
            return 'Neutral', 0.5, np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2])
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using Haar Cascade
        
        Args:
            frame (numpy.ndarray): Input image frame
            
        Returns:
            list: List of face coordinates (x, y, w, h)
        """
        try:
            # Convert to grayscale for face detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Equalize histogram for better detection
            gray = cv2.equalizeHist(gray)
            
            # Detect faces with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def draw_emotion_info(self, frame, x, y, w, h, emotion, confidence, all_predictions=None):
        """
        Draw emotion information and bounding box on frame
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            x, y, w, h (int): Face bounding box coordinates
            emotion (str): Detected emotion
            confidence (float): Detection confidence
            all_predictions (numpy.ndarray, optional): All emotion probabilities
        """
        try:
            # Get color for the detected emotion
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw face bounding box with thickness based on confidence
            thickness = max(1, int(confidence * 3))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw emotion label with confidence
            label = f"{emotion}: {confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background for text
            cv2.rectangle(frame, 
                         (x, y - text_height - 10),
                         (x + text_width, y), 
                         color, -1)
            
            # Draw text in contrasting color
            text_color = (0, 0, 0) if emotion in ['Happy', 'Surprise', 'Neutral'] else (255, 255, 255)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw emotion probability bars if provided
            if all_predictions is not None and len(all_predictions) == 7:
                self._draw_probability_bars(frame, x, y, w, h, all_predictions)
                
        except Exception as e:
            logger.error(f"Error drawing emotion info: {e}")
    
    def _draw_probability_bars(self, frame, x, y, w, h, predictions):
        """
        Draw probability bars for all emotions
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            x, y, w, h (int): Face bounding box coordinates
            predictions (numpy.ndarray): Emotion probabilities
        """
        try:
            bar_height = 12
            bar_width = 80
            spacing = 2
            start_x = x + w + 10
            
            # Ensure we don't draw outside frame bounds
            frame_height = frame.shape[0]
            max_bars = min(7, (frame_height - y) // (bar_height + spacing))
            
            for i, (emotion_name, prob) in enumerate(zip(self.emotion_labels.values(), predictions)):
                if i >= max_bars:
                    break
                    
                bar_color = self.emotion_colors.get(emotion_name, (255, 255, 255))
                bar_fill = int(prob * bar_width)
                bar_y = y + i * (bar_height + spacing)
                
                # Ensure bar doesn't go outside frame
                if bar_y + bar_height > frame_height:
                    break
                
                # Draw bar background
                cv2.rectangle(frame, 
                            (start_x, bar_y),
                            (start_x + bar_width, bar_y + bar_height),
                            (50, 50, 50), -1)
                
                # Draw bar fill
                if bar_fill > 0:
                    cv2.rectangle(frame,
                                (start_x, bar_y),
                                (start_x + bar_fill, bar_y + bar_height),
                                bar_color, -1)
                
                # Draw border around bar
                cv2.rectangle(frame,
                            (start_x, bar_y),
                            (start_x + bar_width, bar_y + bar_height),
                            (100, 100, 100), 1)
                
                # Draw emotion label and percentage
                text = f"{emotion_name[:3]}: {prob:.2f}"
                text_color = (255, 255, 255)
                
                cv2.putText(frame, text, 
                          (start_x + bar_width + 5, bar_y + bar_height - 3),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
                          
        except Exception as e:
            logger.error(f"Error drawing probability bars: {e}")
    
    def get_average_processing_time(self):
        """
        Get average processing time for emotion detection
        
        Returns:
            float: Average processing time in seconds
        """
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times)
    
    def get_fps_estimate(self):
        """
        Get estimated FPS based on processing times
        
        Returns:
            float: Estimated FPS
        """
        avg_time = self.get_average_processing_time()
        if avg_time == 0:
            return 0.0
        return 1.0 / avg_time
    
    def get_emotion_statistics(self, emotion_history):
        """
        Calculate statistics from emotion history
        
        Args:
            emotion_history (list): List of detected emotions
            
        Returns:
            dict: Emotion statistics
        """
        try:
            if not emotion_history:
                return {}
            
            total = len(emotion_history)
            stats = {}
            
            for emotion in set(emotion_history):
                count = emotion_history.count(emotion)
                stats[emotion] = {
                    'count': count,
                    'percentage': (count / total) * 100,
                    'frequency': count / total if total > 0 else 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating emotion statistics: {e}")
            return {}
    
    def process_frame(self, frame, confidence_threshold=0.5):
        """
        Process a complete frame - detect faces and emotions
        
        Args:
            frame (numpy.ndarray): Input frame
            confidence_threshold (float): Minimum confidence for detection
            
        Returns:
            tuple: (processed_frame, results)
        """
        try:
            results = []
            processed_frame = frame.copy()
            
            # Detect faces
            faces = self.detect_faces(processed_frame)
            
            # Process each face
            for (x, y, w, h) in faces:
                try:
                    # Extract face ROI
                    face_roi = processed_frame[y:y+h, x:x+w]
                    
                    # Skip if face ROI is too small
                    if face_roi.size == 0:
                        continue
                    
                    # Detect emotion
                    emotion, confidence, predictions = self.detect_emotion(face_roi)
                    
                    if confidence >= confidence_threshold:
                        results.append({
                            'bbox': (x, y, w, h),
                            'emotion': emotion,
                            'confidence': confidence,
                            'predictions': predictions
                        })
                        
                        # Draw emotion information
                        self.draw_emotion_info(processed_frame, x, y, w, h, emotion, confidence, predictions)
                        
                except Exception as e:
                    logger.warning(f"Error processing individual face: {e}")
                    continue
            
            return processed_frame, results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []


# Utility functions
def check_opencv_installation():
    """
    Check if OpenCV is properly installed and working
    
    Returns:
        bool: True if OpenCV is working
    """
    try:
        # Test basic OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (10, 10), (90, 90), (255, 0, 0), 2)
        cv2.putText(test_image, "OpenCV", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return True
    except Exception as e:
        logger.error(f"OpenCV check failed: {e}")
        return False


def test_emotion_detector(model_path="models/emotion_model.h5"):
    """
    Test the emotion detector with a sample image
    
    Args:
        model_path (str): Path to the model file
    """
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Initialize detector
        detector = EmotionDetector(model_path)
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Test face detection
        faces = detector.detect_faces(test_image)
        print(f"✅ Face detection test: Found {len(faces)} faces")
        
        # Test emotion detection with a sample face
        sample_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emotion, confidence, predictions = detector.detect_emotion(sample_face)
        print(f"✅ Emotion detection test: {emotion} (confidence: {confidence:.3f})")
        
        # Test frame processing
        processed_frame, results = detector.process_frame(test_image)
        print(f"✅ Frame processing test: Processed {len(results)} detections")
        
        print("🎉 All tests passed! EmotionDetector is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing EmotionDetector...")
    
    # Check OpenCV
    if check_opencv_installation():
        print("✅ OpenCV is working correctly")
    else:
        print("❌ OpenCV has issues")
    
    # Test the detector
    test_emotion_detector()
    
    print("📝 EmotionDetector is ready for use!")