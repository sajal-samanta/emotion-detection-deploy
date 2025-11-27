import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tempfile
import os
import time
import json
from datetime import datetime
from io import BytesIO

# Add src to path and import custom modules
import sys
sys.path.append('src')

try:
    from utils import EmotionDetector
except ImportError as e:
    st.error(f"Error importing custom modules: {e}")
    st.info("Please ensure the 'src' directory contains utils.py")

# Page configuration
st.set_page_config(
    page_title="AI Emotion Detector Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2.2rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1rem 0;
        font-weight: 700;
    }
    .webcam-container {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid #4ECDC4;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .feature-card {
        background: linear-gradient(135deg, #2d3746, #1a202c);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        color: white;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
    }
    .emotion-result-card {
        background: linear-gradient(135deg, #1e1e1e, #2d2d2d);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .emotion-result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    .confidence-bar {
        height: 8px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4ecdc4, #45b7d1);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        background: linear-gradient(135deg, #4ECDC4, #44A08D);
        color: white;
        border: none;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.4);
        background: linear-gradient(135deg, #44A08D, #4ECDC4);
    }
    .tab-content {
        padding: 2rem;
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border-radius: 15px;
        margin: 1rem 0;
    }
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 600;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

class SimpleWebcamProcessor:
    """Simplified webcam processor that works with Streamlit"""
    def __init__(self, emotion_detector):
        self.detector = emotion_detector
        self.cap = None
        self.is_running = False
        
    def start_camera(self):
        """Start webcam capture"""
        try:
            # Release any existing camera
            if self.cap:
                self.cap.release()
                
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    # Test if camera actually works
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Camera {camera_index} working!")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    self.cap = None
            
            if self.cap is None or not self.cap.isOpened():
                st.error("❌ Could not access any webcam. Please check:")
                st.error("1. Webcam is connected and not being used by another application")
                st.error("2. Camera permissions are granted")
                st.error("3. Try refreshing the page")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            return True
            
        except Exception as e:
            st.error(f"❌ Webcam error: {str(e)}")
            return False
    
    def get_frame(self):
        """Get current frame from webcam"""
        try:
            if self.cap and self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
            return None
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame with emotion detection"""
        if frame is None:
            return None
            
        try:
            # Create a copy to avoid modifying original
            processed_frame = frame.copy()
            
            # Detect faces
            faces = self.detector.detect_faces(processed_frame)
            
            # Track emotions for statistics
            frame_emotions = []
            
            # Process each face
            for (x, y, w, h) in faces:
                try:
                    # Extract face ROI
                    face_roi = processed_frame[y:y+h, x:x+w]
                    
                    if face_roi.size == 0:
                        continue
                    
                    # Detect emotion
                    emotion, confidence, predictions = self.detector.detect_emotion(face_roi)
                    
                    # Only draw if confidence is reasonable
                    if confidence > 0.3:
                        # Draw emotion information
                        self.detector.draw_emotion_info(processed_frame, x, y, w, h, emotion, confidence)
                        frame_emotions.append((emotion, confidence))
                        
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
                    
            return processed_frame, frame_emotions
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, []
    
    def stop_camera(self):
        """Stop webcam capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

class EmotionDetectionApp:
    def __init__(self):
        self.model_paths = [
            "models/emotion_model.h5",
            "models/emotion_model.keras",
            "emotion_model.h5", 
            "emotion_model.keras"
        ]
        self.detector = None
        self.webcam_processor = None
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 0.5
        if 'emotion_history' not in st.session_state:
            st.session_state.emotion_history = []
        if 'detection_count' not in st.session_state:
            st.session_state.detection_count = 0
        if 'webcam_initialized' not in st.session_state:
            st.session_state.webcam_initialized = False
        if 'last_emotion' not in st.session_state:
            st.session_state.last_emotion = "Neutral"
        if 'last_confidence' not in st.session_state:
            st.session_state.last_confidence = 0.0
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Description"
    
    def find_model(self):
        """Find the model file"""
        for path in self.model_paths:
            if os.path.exists(path):
                return path
        return None
    
    def load_model(self):
        """Load emotion detection model"""
        try:
            if self.detector is None:
                model_path = self.find_model()
                if model_path:
                    self.detector = EmotionDetector(model_path)
                    return True
                else:
                    st.error("❌ Model file not found!")
                    st.info("Please train the model first using the Jupyter notebook")
                    return False
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def render_navigation(self):
        """Render navigation tabs"""
        st.sidebar.markdown("## 🧭 Navigation")
        
        # Navigation tabs
        pages = {
            "📖 Description": "Description",
            "🎭 Emotion Detection": "Detection",
            "📊 Analytics": "Analytics"
        }
        
        selected_page = st.sidebar.radio(
            "Go to",
            list(pages.keys()),
            index=0
        )
        
        st.session_state.current_page = pages[selected_page]
        
        st.sidebar.markdown("---")
        
        # Model Status
        st.sidebar.markdown("### 🔧 Model Status")
        model_path = self.find_model()
        if model_path:
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            st.sidebar.success(f"✅ Model Loaded\nSize: {model_size:.1f} MB")
        else:
            st.sidebar.error("❌ Model Not Found")
        
        return pages[selected_page]
    
    def render_description_page(self):
        """Render the description/introduction page"""
        st.markdown('<h1 class="main-header">🧠 AI Emotion Detector Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.3rem; margin-bottom: 3rem;">Advanced Deep Learning for Real-time Facial Emotion Recognition</p>', unsafe_allow_html=True)
        
        # Hero Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style='padding: 2rem; background: linear-gradient(135deg, #1a1a1a, #2d2d2d); 
                        border-radius: 20px; border-left: 5px solid #4ECDC4;'>
                <h2 style='color: #4ECDC4; margin-bottom: 1rem;'>🎯 Revolutionizing Emotion Analysis</h2>
                <p style='font-size: 1.1rem; line-height: 1.6; color: #e0e0e0;'>
                Welcome to the next generation of emotion recognition technology. Our AI-powered system 
                leverages state-of-the-art deep learning to accurately detect and analyze human emotions 
                in real-time through facial expressions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <div class="stat-number">68%</div>
                <p style='color: #6c757d;'>Accuracy Rate</p>
                <div class="stat-number">7</div>
                <p style='color: #6c757d;'>Emotions Detected</p>
                <div class="stat-number">Real-time</div>
                <p style='color: #6c757d;'>Processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features Grid
        st.markdown('<h2 class="section-header">🚀 Key Features</h2>', unsafe_allow_html=True)
        
        features = [
            {
                "icon": "🧠",
                "title": "Advanced CNN Architecture",
                "description": "Deep neural network trained on FER2013 dataset with 68% accuracy",
                "color": "#4ECDC4"
            },
            {
                "icon": "⚡",
                "title": "Real-time Processing",
                "description": "Live emotion detection at 30 FPS with instant visualization",
                "color": "#45B7D1"
            },
            {
                "icon": "📊",
                "title": "Comprehensive Analytics",
                "description": "Detailed emotion statistics, trends, and distribution analysis",
                "color": "#FF6B6B"
            },
            {
                "icon": "🎯",
                "title": "Multi-Modal Detection",
                "description": "Support for webcam, images, and batch processing",
                "color": "#667eea"
            },
            {
                "icon": "🔬",
                "title": "Scientific Accuracy",
                "description": "Based on psychological research and validated emotion models",
                "color": "#764ba2"
            },
            {
                "icon": "💾",
                "title": "Data Export",
                "description": "Export results in JSON, CSV, and image formats for analysis",
                "color": "#FFA62B"
            }
        ]
        
        # Create feature cards in a grid
        cols = st.columns(3)
        for idx, feature in enumerate(features):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="feature-card" style="border-left-color: {feature['color']}">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">{feature['icon']}</div>
                    <h4 style="color: {feature['color']}; margin-bottom: 0.5rem;">{feature['title']}</h4>
                    <p style="color: #b0b0b0; line-height: 1.5;">{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Technology Stack
        st.markdown('<h2 class="section-header">🛠️ Technology Stack</h2>', unsafe_allow_html=True)
        
        tech_stack = [
            "TensorFlow & Keras", "OpenCV", "Deep Learning", "CNN Architecture",
            "Streamlit", "Plotly", "NumPy", "Pandas", "Computer Vision",
            "Haar Cascades", "Real-time Processing", "Data Visualization"
        ]
        
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
        """ + "".join([f'<span class="tech-badge">{tech}</span>' for tech in tech_stack]) + """
        </div>
        """, unsafe_allow_html=True)
        
        # Getting Started
        st.markdown("---")
        st.markdown('<h2 class="section-header">🚀 Getting Started</h2>', unsafe_allow_html=True)
        
        steps = [
            {"step": "1", "title": "Navigate to Detection", "desc": "Click on 'Emotion Detection' in the sidebar"},
            {"step": "2", "title": "Choose Input Mode", "desc": "Select between Live Webcam or Image Upload"},
            {"step": "3", "title": "Start Detection", "desc": "Click start and watch real-time emotion analysis"},
            {"step": "4", "title": "Analyze Results", "desc": "View detailed analytics and export your data"}
        ]
        
        step_cols = st.columns(4)
        for idx, step in enumerate(steps):
            with step_cols[idx]:
                st.markdown(f"""
                <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #2d2d2d, #1a1a1a); 
                            border-radius: 15px; border: 1px solid #333;'>
                    <div style='font-size: 2rem; color: #4ECDC4; margin-bottom: 0.5rem;'>{step['step']}</div>
                    <h4 style='color: white; margin-bottom: 0.5rem;'>{step['title']}</h4>
                    <p style='color: #b0b0b0; font-size: 0.9rem;'>{step['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Call to Action
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3 style='color: #4ECDC4; margin-bottom: 1rem;'>Ready to Explore Emotions?</h3>
                <p style='color: #b0b0b0; margin-bottom: 2rem;'>
                Experience the power of AI-driven emotion recognition. Click the button below to start detecting emotions in real-time!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎭 Start Emotion Detection →", use_container_width=True, type="primary"):
                st.session_state.current_page = "Detection"
                st.rerun()
    
    def render_detection_sidebar(self):
        """Render sidebar for detection page"""
        st.sidebar.markdown("### ⚙️ Detection Settings")
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            0.1, 1.0, 0.5, 0.05,
            help="Adjust sensitivity of emotion detection"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Live Statistics")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Detections", st.session_state.detection_count)
        with col2:
            st.metric("Unique Emotions", len(set(st.session_state.emotion_history)))
        
        st.sidebar.markdown(f"**Current Emotion:** {st.session_state.last_emotion}")
        st.sidebar.markdown(f"**Confidence:** {st.session_state.last_confidence:.2f}")
        
        return confidence_threshold
    
    def render_detection_page(self):
        """Render the main detection page"""
        st.markdown('<h1 class="section-header">🎭 Emotion Detection</h1>', unsafe_allow_html=True)
        
        # Model check
        if not self.load_model():
            st.error("Please train the model first")
            return
        
        # Detection mode selection
        detection_mode = st.radio(
            "Select Detection Mode:",
            ["📷 Live Webcam Detection", "🖼️ Image Upload"],
            horizontal=True,
            key="detection_mode"
        )
        
        confidence_threshold = self.render_detection_sidebar()
        st.session_state.confidence_threshold = confidence_threshold
        
        st.markdown("---")
        
        if "Live Webcam" in detection_mode:
            self.render_webcam_interface()
        else:
            self.render_image_upload_interface()
    
    def render_professional_results(self, results):
        """Render professional-looking detection results"""
        st.markdown("### 📋 Detection Results")
        
        if not results:
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #2d2d2d, #1a1a1a); 
                        border-radius: 15px; border: 2px dashed #444;'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>🔍</div>
                <h3 style='color: #6c757d; margin-bottom: 1rem;'>No Emotions Detected</h3>
                <p style='color: #8a8a8a;'>Try adjusting the confidence threshold or ensure faces are clearly visible.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Summary card
        total_faces = len(results)
        emotions_detected = list(set([r['emotion'] for r in results]))
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 1.5rem; 
                    border-radius: 15px; color: white; margin-bottom: 2rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h3 style='margin: 0; color: white;'>Detection Summary</h3>
                    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
                    {total_faces} face(s) detected • {len(emotions_detected)} emotion(s) • {avg_confidence:.1%} avg confidence
                    </p>
                </div>
                <div style='font-size: 2.5rem;'>🎯</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual face results
        for i, result in enumerate(results):
            emotion_color = self.detector.emotion_colors.get(result['emotion'], (255, 255, 255))
            color_hex = '#%02x%02x%02x' % emotion_color[::-1]
            confidence = result['confidence']
            emotion_emoji = self.get_emotion_emoji(result['emotion'])
            
            # Create the emotion probability bars HTML
            probability_bars = self.render_emotion_probability_bars(result['predictions'])
            
            st.markdown(f"""
            <div class="emotion-result-card" style="border-color: {color_hex}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                            <div style="font-size: 2rem;">{emotion_emoji}</div>
                            <div>
                                <h3 style="margin: 0; color: {color_hex}; font-size: 1.5rem;">
                                    Face {i+1}: {result['emotion']}
                                </h3>
                                <p style="margin: 0.2rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">
                                    Confidence Level: <strong>{confidence:.3f}</strong>
                                </p>
                            </div>
                        </div>
                        
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #8a8a8a;">
                            <span>0%</span>
                            <span>{confidence * 100:.1f}%</span>
                            <span>100%</span>
                        </div>
                    </div>
                    
                    <div style="background: {color_hex}20; padding: 0.8rem; border-radius: 10px; 
                                text-align: center; min-width: 80px;">
                        <div style="font-size: 1.8rem; font-weight: 800; color: {color_hex};">
                            {confidence * 100:.0f}%
                        </div>
                        <div style="font-size: 0.7rem; color: {color_hex}; opacity: 0.8;">
                            CONFIDENCE
                        </div>
                    </div>
                </div>
                
                <!-- Emotion Probability Distribution -->
                <div style="margin-top: 1rem;">
                    <p style="margin: 0 0 0.5rem 0; color: #b0b0b0; font-size: 0.9rem;">
                        <strong>Emotion Probability Distribution:</strong>
                    </p>
                    {probability_bars}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_emotion_probability_bars(self, predictions):
        """Render emotion probability bars"""
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        bars_html = ""
        
        for emotion, prob in zip(emotions, predictions):
            color = self.detector.emotion_colors.get(emotion, (255, 255, 255))
            color_hex = '#%02x%02x%02x' % color[::-1]
            width = max(2, prob * 100)  # Minimum 2% width for visibility
            
            bars_html += f"""
            <div style="display: flex; align-items: center; margin: 0.3rem 0; gap: 1rem;">
                <div style="min-width: 80px; color: #e0e0e0; font-size: 0.9rem;">{emotion}</div>
                <div style="flex: 1; background: #2d2d2d; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="width: {width}%; height: 100%; background: {color_hex}; border-radius: 4px;"></div>
                </div>
                <div style="min-width: 40px; text-align: right; color: #b0b0b0; font-size: 0.8rem;">
                    {prob:.1%}
                </div>
            </div>
            """
        
        return bars_html
    
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion"""
        emoji_map = {
            'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
            'Happy': '😊', 'Sad': '😢', 'Surprise': '😲',
            'Neutral': '😐'
        }
        return emoji_map.get(emotion, '❓')
    
    def render_webcam_interface(self):
        """Render live webcam interface"""
        st.markdown("### 📷 Live Webcam Emotion Detection")
        
        # Initialize webcam processor if needed
        if st.session_state.webcam_running and self.webcam_processor is None:
            if self.load_model():
                self.webcam_processor = SimpleWebcamProcessor(self.detector)
                if not self.webcam_processor.start_camera():
                    st.session_state.webcam_running = False
                    st.error("❌ Failed to start webcam")
        
        # Webcam status and controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not st.session_state.webcam_running:
                st.markdown("""
                <div class='webcam-container'>
                    <h3>🚀 Ready to Start</h3>
                    <p>Click the button below to start real-time emotion detection</p>
                    <p><small>Make sure your webcam is connected and you've granted camera permissions</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🎥 START WEBCAM", use_container_width=True, type="primary"):
                    if self.load_model():
                        self.webcam_processor = SimpleWebcamProcessor(self.detector)
                        if self.webcam_processor.start_camera():
                            st.session_state.webcam_running = True
                            st.session_state.webcam_initialized = True
                            st.rerun()
                        else:
                            st.error("❌ Could not start webcam")
            else:
                st.markdown("""
                <div class='webcam-container'>
                    <h3>🔴 LIVE - Webcam Active</h3>
                    <p>Real-time emotion detection is running</p>
                    <p><small>Make facial expressions to see emotion detection in action!</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("⏹️ STOP WEBCAM", use_container_width=True, type="secondary"):
                    self.stop_webcam()
                    st.session_state.webcam_running = False
                    st.rerun()
        
        with col2:
            # Quick stats
            st.markdown("""
            <div style='background: linear-gradient(135deg, #2d3746, #1a202c); 
                        padding: 1.5rem; border-radius: 15px; color: white;'>
                <h4>📈 Live Stats</h4>
                <p><strong>Detections:</strong> {}</p>
                <p><strong>Unique Emotions:</strong> {}</p>
                <p><strong>Current Emotion:</strong> {}</p>
                <p><strong>Confidence:</strong> {:.2f}</p>
                <p><strong>Status:</strong> {}</p>
            </div>
            """.format(
                st.session_state.detection_count,
                len(set(st.session_state.emotion_history)),
                st.session_state.last_emotion,
                st.session_state.last_confidence,
                "🔴 Active" if st.session_state.webcam_running else "⏸️ Ready"
            ), unsafe_allow_html=True)
        
        # Webcam preview
        if st.session_state.webcam_running:
            self.render_webcam_preview()
        else:
            self.render_webcam_instructions()
        
        # Analytics (show even when webcam is not running if we have data)
        if st.session_state.emotion_history:
            self.render_analytics()
    
    def stop_webcam(self):
        """Stop webcam processing"""
        if self.webcam_processor:
            self.webcam_processor.stop_camera()
            self.webcam_processor = None
        st.session_state.webcam_initialized = False
    
    def render_webcam_preview(self):
        """Render webcam preview with real-time processing"""
        st.markdown("### 👁️ Live Preview")
        
        if self.webcam_processor is None or not self.webcam_processor.is_running:
            st.error("❌ Webcam not available. Please restart the webcam.")
            return
        
        # Create a placeholder for the webcam feed
        preview_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Display webcam feed for a limited time to prevent freezing
        max_frames = 300  # Show max 300 frames (~10 seconds at 30fps)
        frame_count = 0
        
        with status_placeholder:
            st.info("🔄 Starting webcam feed... Make sure your face is visible!")
        
        while (st.session_state.webcam_running and 
               self.webcam_processor and 
               self.webcam_processor.is_running and
               frame_count < max_frames):
            
            try:
                # Get frame from webcam
                frame = self.webcam_processor.get_frame()
                
                if frame is not None:
                    # Process frame (detect emotions)
                    processed_frame, frame_emotions = self.webcam_processor.process_frame(frame)
                    
                    if processed_frame is not None:
                        # Convert to RGB for display
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        preview_placeholder.image(
                            frame_rgb, 
                            channels="RGB", 
                            use_container_width=True,
                            caption="🎭 Live Emotion Detection - Make expressions to see emotions!"
                        )
                        
                        # Update emotion history and stats
                        for emotion, confidence in frame_emotions:
                            st.session_state.emotion_history.append(emotion)
                            st.session_state.detection_count += 1
                        
                        if frame_emotions:
                            st.session_state.last_emotion = frame_emotions[0][0]
                            st.session_state.last_confidence = frame_emotions[0][1]
                        
                        frame_count += 1
                        
                        # Update status
                        with status_placeholder:
                            st.success(f"✅ Live feed active - Frame {frame_count}/{max_frames}")
                
                # Small delay
                time.sleep(0.05)  # ~20 FPS for stability
                
            except Exception as e:
                with status_placeholder:
                    st.error(f"❌ Webcam error: {str(e)}")
                break
        
        # Show completion message
        if frame_count >= max_frames:
            with status_placeholder:
                st.warning("⏸️ Preview completed. Click 'START WEBCAM' to continue.")
            self.stop_webcam()
            st.session_state.webcam_running = False
            st.rerun()
    
    def render_webcam_instructions(self):
        """Render webcam setup instructions"""
        st.markdown("""
        <div class='feature-card'>
            <h3>🎯 Webcam Setup Guide</h3>
            <div style='margin: 1rem 0;'>
                <h4>📝 Step-by-Step Instructions:</h4>
                <ol>
                    <li><strong>Check webcam connection</strong> - Ensure your webcam is properly connected</li>
                    <li><strong>Grant permissions</strong> - Allow camera access when browser prompts</li>
                    <li><strong>Click "START WEBCAM"</strong> - Begin real-time detection</li>
                    <li><strong>Position yourself</strong> - Face the camera directly in good lighting</li>
                    <li><strong>Make expressions</strong> - Show different emotions to test detection</li>
                </ol>
            </div>
            <div style='margin: 1rem 0;'>
                <h4>💡 Pro Tips for Best Results:</h4>
                <ul>
                    <li>✅ Use natural daylight or well-lit room</li>
                    <li>✅ Avoid backlighting from windows</li>
                    <li>✅ Keep your face clearly visible</li>
                    <li>✅ Maintain neutral background</li>
                    <li>✅ Stay within 1-2 meters from camera</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_image_upload_interface(self):
        """Render image upload interface"""
        st.markdown("### 📸 Image Emotion Detection")
        
        uploaded_file = st.file_uploader(
            "Upload an image for emotion analysis",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="🖼️ Original Image", use_container_width=True)
            
            with col2:
                st.info(f"**Image Details:**\n- Size: {image.size[0]}x{image.size[1]}\n- Mode: {image.mode}")
                
                if st.button("🔍 Analyze Emotions", type="primary", use_container_width=True):
                    if self.load_model():
                        with st.spinner("🔄 Detecting emotions..."):
                            processed_image, results = self.process_image(image)
                        
                        if processed_image is not None:
                            st.image(processed_image, caption="🎭 Emotion Analysis Result", use_container_width=True)
                            
                            if results:
                                self.render_professional_results(results)
                            else:
                                st.warning("❌ No faces detected or emotions below confidence threshold")
    
    def process_image(self, image):
        """Process uploaded image for emotion detection"""
        try:
            # Convert PIL to OpenCV
            image_cv = np.array(image)
            if image_cv.shape[-1] == 4:  # RGBA
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            # Detect faces and emotions
            faces = self.detector.detect_faces(image_cv)
            results = []
            
            for (x, y, w, h) in faces:
                face_roi = image_cv[y:y+h, x:x+w]
                emotion, confidence, predictions = self.detector.detect_emotion(face_roi)
                
                if confidence >= st.session_state.confidence_threshold:
                    results.append({
                        'bbox': (x, y, w, h),
                        'emotion': emotion,
                        'confidence': confidence,
                        'predictions': predictions
                    })
                    
                    # Draw on image
                    self.detector.draw_emotion_info(image_cv, x, y, w, h, emotion, confidence)
                    
                    # Update statistics
                    st.session_state.detection_count += 1
                    st.session_state.emotion_history.append(emotion)
                    st.session_state.last_emotion = emotion
                    st.session_state.last_confidence = confidence
            
            # Convert back to RGB
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            return image_rgb, results
            
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            return None, []
    
    def render_analytics(self):
        """Render emotion analytics"""
        if not st.session_state.emotion_history:
            return
            
        st.markdown("### 📊 Emotion Analytics")
        
        # Emotion distribution
        emotion_series = pd.Series(st.session_state.emotion_history)
        emotion_counts = emotion_series.value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                title="📈 Emotion Distribution",
                color=emotion_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3,
                labels={'x': 'Emotion', 'y': 'Count'}
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            if len(emotion_counts) > 0:
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="🥧 Emotion Proportions",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No emotion data to display")
    
    def render_analytics_page(self):
        """Render analytics page"""
        st.markdown('<h1 class="section-header">📊 Emotion Analytics</h1>', unsafe_allow_html=True)
        
        if not st.session_state.emotion_history:
            st.info("📈 No analytics data available yet. Start detecting emotions to see analytics!")
            return
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{len(st.session_state.emotion_history)}</div>
                <p>Total Detections</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{len(set(st.session_state.emotion_history))}</div>
                <p>Unique Emotions</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            dominant = max(set(st.session_state.emotion_history), key=st.session_state.emotion_history.count)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{self.get_emotion_emoji(dominant)}</div>
                <p>Dominant Emotion</p>
                <p style="font-size: 1.2rem; font-weight: 600; margin: 0;">{dominant}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{st.session_state.detection_count}</div>
                <p>Total Faces</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotion distribution
            emotion_series = pd.Series(st.session_state.emotion_history)
            emotion_counts = emotion_series.value_counts()
            
            fig_bar = px.bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                title="📈 Emotion Distribution",
                color=emotion_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3,
                labels={'x': 'Emotion', 'y': 'Count'}
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="🥧 Emotion Proportions",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_main_interface(self):
        """Render main application interface based on current page"""
        current_page = self.render_navigation()
        
        if current_page == "Description":
            self.render_description_page()
        elif current_page == "Detection":
            self.render_detection_page()
        elif current_page == "Analytics":
            self.render_analytics_page()

def main():
    # Add loading animation
    with st.spinner("🚀 Loading AI Emotion Detector Pro..."):
        time.sleep(1)
    
    app = EmotionDetectionApp()
    app.render_main_interface()

if __name__ == "__main__":
    main()