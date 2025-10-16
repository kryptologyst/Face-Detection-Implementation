"""
Streamlit web interface for face detection demo.

This module provides a user-friendly web interface for testing different
face detection methods and visualizing results.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import time

# Import our custom modules
from src.face_detector import FaceDetector, FaceDetectionResult, create_sample_image_with_faces
from src.config_manager import config_manager


class FaceDetectionApp:
    """Main Streamlit application for face detection."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = config_manager.config
        self.setup_logging()
        self.setup_page_config()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_page_config(self) -> None:
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title=self.config.ui.title,
            page_icon="👤",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self) -> None:
        """Run the main application."""
        st.title("🔍 Face Detection Demo")
        st.markdown("---")
        
        # Sidebar for configuration
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_input_section()
        
        with col2:
            self.render_output_section()
        
        # Additional features
        self.render_additional_features()
    
    def render_sidebar(self) -> None:
        """Render the sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # Detection method selection
        available_methods = ["opencv"]
        
        # Check for optional dependencies
        try:
            from mtcnn import MTCNN
            available_methods.append("mtcnn")
        except ImportError:
            pass
        
        try:
            import mediapipe as mp
            available_methods.append("mediapipe")
        except ImportError:
            pass
        
        try:
            import torch
            from facenet_pytorch import MTCNN as FacenetMTCNN
            available_methods.append("facenet")
        except ImportError:
            pass
        
        selected_method = st.sidebar.selectbox(
            "Detection Method",
            available_methods,
            index=available_methods.index(self.config.detection.method)
        )
        
        # Confidence threshold
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=self.config.detection.confidence_threshold,
            step=0.05
        )
        
        # Additional parameters for OpenCV
        if selected_method == "opencv":
            scale_factor = st.sidebar.slider(
                "Scale Factor",
                min_value=1.01,
                max_value=2.0,
                value=self.config.detection.scale_factor,
                step=0.01
            )
            
            min_neighbors = st.sidebar.slider(
                "Min Neighbors",
                min_value=1,
                max_value=20,
                value=self.config.detection.min_neighbors
            )
        
        st.sidebar.markdown("---")
        
        # Method descriptions
        st.sidebar.markdown("### 📖 Method Descriptions")
        method_descriptions = {
            "opencv": "**OpenCV Haar Cascades**: Fast and lightweight, good for real-time applications",
            "mtcnn": "**MTCNN**: Multi-task CNN with high accuracy and landmark detection",
            "mediapipe": "**MediaPipe**: Google's solution with good balance of speed and accuracy",
            "facenet": "**FaceNet MTCNN**: PyTorch implementation with state-of-the-art accuracy"
        }
        
        for method, description in method_descriptions.items():
            if method in available_methods:
                st.sidebar.markdown(f"**{method.upper()}**: {description}")
    
    def render_input_section(self) -> None:
        """Render the input section."""
        st.header("📸 Input Image")
        
        # Image upload options
        input_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Sample Image", "Webcam"]
        )
        
        image = None
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an image containing faces to detect"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = np.array(image)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        elif input_option == "Use Sample Image":
            if st.button("Generate Sample Image"):
                image = create_sample_image_with_faces()
                st.session_state.sample_image = image
        
        elif input_option == "Webcam":
            if st.button("Capture from Webcam"):
                # Note: This would require additional setup for webcam access
                st.warning("Webcam functionality requires additional setup. Please upload an image instead.")
        
        # Display input image
        if image is not None:
            st.image(image, caption="Input Image", use_column_width=True)
            st.session_state.input_image = image
        elif 'sample_image' in st.session_state:
            st.image(st.session_state.sample_image, caption="Sample Image", use_column_width=True)
            st.session_state.input_image = st.session_state.sample_image
    
    def render_output_section(self) -> None:
        """Render the output section."""
        st.header("🎯 Detection Results")
        
        if 'input_image' not in st.session_state:
            st.info("Please provide an input image first.")
            return
        
        # Get configuration from sidebar
        selected_method = st.sidebar.selectbox("Detection Method", ["opencv", "mtcnn", "mediapipe", "facenet"])
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        
        # Detect faces
        if st.button("🔍 Detect Faces", type="primary"):
            with st.spinner("Detecting faces..."):
                try:
                    detector = FaceDetector(
                        method=selected_method,
                        confidence_threshold=confidence_threshold
                    )
                    
                    start_time = time.time()
                    result = detector.detect_faces(st.session_state.input_image)
                    detection_time = time.time() - start_time
                    
                    # Draw detections
                    output_image = detector.draw_detections(st.session_state.input_image, result)
                    
                    # Display results
                    st.image(output_image, caption=f"Detection Results ({selected_method.upper()})", use_column_width=True)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Faces Detected", result.num_faces)
                    with col2:
                        st.metric("Detection Time", f"{detection_time:.3f}s")
                    with col3:
                        st.metric("Method", selected_method.upper())
                    
                    # Show confidence scores if available
                    if result.confidence_scores:
                        st.subheader("Confidence Scores")
                        for i, score in enumerate(result.confidence_scores):
                            st.progress(score, text=f"Face {i+1}: {score:.3f}")
                    
                    # Store results in session state
                    st.session_state.last_result = result
                    st.session_state.last_output_image = output_image
                    
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    st.exception(e)
        
        # Show previous results if available
        if 'last_result' in st.session_state:
            st.subheader("📊 Detection Details")
            result = st.session_state.last_result
            
            # Create a detailed results table
            if result.bounding_boxes:
                import pandas as pd
                
                data = []
                for i, (x, y, w, h) in enumerate(result.bounding_boxes):
                    row = {
                        "Face": i + 1,
                        "X": x,
                        "Y": y,
                        "Width": w,
                        "Height": h,
                        "Area": w * h
                    }
                    if i < len(result.confidence_scores):
                        row["Confidence"] = f"{result.confidence_scores[i]:.3f}"
                    data.append(row)
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
    
    def render_additional_features(self) -> None:
        """Render additional features section."""
        st.markdown("---")
        st.header("🔧 Additional Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Performance Comparison")
            if st.button("Compare All Methods"):
                self.compare_methods()
        
        with col2:
            st.subheader("💾 Export Results")
            if 'last_output_image' in st.session_state:
                # Convert BGR to RGB for PIL
                output_rgb = cv2.cvtColor(st.session_state.last_output_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(output_rgb)
                
                # Create download button
                import io
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Result Image",
                    data=byte_im,
                    file_name="face_detection_result.png",
                    mime="image/png"
                )
    
    def compare_methods(self) -> None:
        """Compare different detection methods."""
        if 'input_image' not in st.session_state:
            st.warning("Please provide an input image first.")
            return
        
        st.subheader("🆚 Method Comparison")
        
        available_methods = ["opencv"]
        
        # Check for optional dependencies
        try:
            from mtcnn import MTCNN
            available_methods.append("mtcnn")
        except ImportError:
            pass
        
        try:
            import mediapipe as mp
            available_methods.append("mediapipe")
        except ImportError:
            pass
        
        try:
            import torch
            from facenet_pytorch import MTCNN as FacenetMTCNN
            available_methods.append("facenet")
        except ImportError:
            pass
        
        results_data = []
        
        for method in available_methods:
            try:
                with st.spinner(f"Testing {method.upper()}..."):
                    detector = FaceDetector(method=method)
                    start_time = time.time()
                    result = detector.detect_faces(st.session_state.input_image)
                    detection_time = time.time() - start_time
                    
                    results_data.append({
                        "Method": method.upper(),
                        "Faces": result.num_faces,
                        "Time (s)": f"{detection_time:.3f}",
                        "Avg Confidence": f"{np.mean(result.confidence_scores):.3f}" if result.confidence_scores else "N/A"
                    })
                    
            except Exception as e:
                results_data.append({
                    "Method": method.upper(),
                    "Faces": "Error",
                    "Time (s)": "N/A",
                    "Avg Confidence": "N/A"
                })
        
        # Display comparison table
        import pandas as pd
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)


def main():
    """Main entry point for the Streamlit app."""
    app = FaceDetectionApp()
    app.run()


if __name__ == "__main__":
    main()
