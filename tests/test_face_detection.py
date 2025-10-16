"""
Test suite for face detection project.

This module contains unit tests and integration tests for the face detection
functionality.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from src.face_detector import FaceDetector, FaceDetectionResult, create_sample_image_with_faces
from src.config_manager import ConfigManager, AppConfig, DetectionConfig, UIConfig, LoggingConfig


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sample_image = create_sample_image_with_faces()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_opencv_detector_initialization(self):
        """Test OpenCV detector initialization."""
        detector = FaceDetector(method="opencv")
        assert detector.method == "opencv"
        assert detector.confidence_threshold == 0.5
        assert detector.detector is not None
    
    def test_opencv_face_detection(self):
        """Test OpenCV face detection."""
        detector = FaceDetector(method="opencv")
        result = detector.detect_faces(self.sample_image)
        
        assert isinstance(result, FaceDetectionResult)
        assert result.method == "opencv"
        assert isinstance(result.bounding_boxes, list)
        assert isinstance(result.num_faces, int)
        assert result.num_faces >= 0
    
    def test_detection_with_file_path(self):
        """Test detection with file path input."""
        # Save sample image to temporary file
        temp_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        cv2.imwrite(temp_image_path, self.sample_image)
        
        detector = FaceDetector(method="opencv")
        result = detector.detect_faces(temp_image_path)
        
        assert isinstance(result, FaceDetectionResult)
        assert result.method == "opencv"
    
    def test_detection_with_path_object(self):
        """Test detection with Path object input."""
        # Save sample image to temporary file
        temp_image_path = Path(self.temp_dir) / "test_image.jpg"
        cv2.imwrite(str(temp_image_path), self.sample_image)
        
        detector = FaceDetector(method="opencv")
        result = detector.detect_faces(temp_image_path)
        
        assert isinstance(result, FaceDetectionResult)
        assert result.method == "opencv"
    
    def test_invalid_image_path(self):
        """Test detection with invalid image path."""
        detector = FaceDetector(method="opencv")
        
        with pytest.raises(ValueError):
            detector.detect_faces("nonexistent_image.jpg")
    
    def test_draw_detections(self):
        """Test drawing detections on image."""
        detector = FaceDetector(method="opencv")
        result = detector.detect_faces(self.sample_image)
        
        output_image = detector.draw_detections(self.sample_image, result)
        
        assert isinstance(output_image, np.ndarray)
        assert output_image.shape == self.sample_image.shape
    
    def test_confidence_threshold(self):
        """Test confidence threshold parameter."""
        detector = FaceDetector(method="opencv", confidence_threshold=0.8)
        assert detector.confidence_threshold == 0.8
    
    def test_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError):
            FaceDetector(method="invalid_method")
    
    @pytest.mark.skipif(not hasattr(cv2, 'data'), reason="OpenCV not available")
    def test_mtcnn_availability_check(self):
        """Test MTCNN availability check."""
        # This test will be skipped if MTCNN is not available
        try:
            detector = FaceDetector(method="mtcnn")
            result = detector.detect_faces(self.sample_image)
            assert isinstance(result, FaceDetectionResult)
        except ImportError:
            pytest.skip("MTCNN not available")


class TestFaceDetectionResult:
    """Test cases for FaceDetectionResult class."""
    
    def test_result_initialization(self):
        """Test FaceDetectionResult initialization."""
        boxes = [(10, 20, 30, 40), (50, 60, 70, 80)]
        scores = [0.9, 0.8]
        landmarks = [[(15, 25), (25, 25)], [(55, 65), (65, 65)]]
        
        result = FaceDetectionResult(
            bounding_boxes=boxes,
            confidence_scores=scores,
            landmarks=landmarks,
            method="test"
        )
        
        assert result.bounding_boxes == boxes
        assert result.confidence_scores == scores
        assert result.landmarks == landmarks
        assert result.method == "test"
        assert result.num_faces == 2
    
    def test_result_without_optional_params(self):
        """Test FaceDetectionResult without optional parameters."""
        boxes = [(10, 20, 30, 40)]
        
        result = FaceDetectionResult(bounding_boxes=boxes, method="test")
        
        assert result.bounding_boxes == boxes
        assert result.confidence_scores == []
        assert result.landmarks == []
        assert result.method == "test"
        assert result.num_faces == 1
    
    def test_result_repr(self):
        """Test FaceDetectionResult string representation."""
        boxes = [(10, 20, 30, 40)]
        result = FaceDetectionResult(bounding_boxes=boxes, method="test")
        
        repr_str = repr(result)
        assert "FaceDetectionResult" in repr_str
        assert "test" in repr_str
        assert "faces=1" in repr_str


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config_manager = ConfigManager("nonexistent_config.yaml")
        config = config_manager.load_config()
        
        assert isinstance(config, AppConfig)
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.detection.method == "opencv"
    
    def test_config_save_and_load(self):
        """Test configuration save and load."""
        config_manager = ConfigManager(self.config_path)
        
        # Create default config
        config = config_manager.load_config()
        
        # Modify config
        config.detection.method = "mtcnn"
        config.detection.confidence_threshold = 0.7
        
        # Save config
        config_manager.save_config(config)
        
        # Load config again
        new_config_manager = ConfigManager(self.config_path)
        loaded_config = new_config_manager.load_config()
        
        assert loaded_config.detection.method == "mtcnn"
        assert loaded_config.detection.confidence_threshold == 0.7
    
    def test_config_property(self):
        """Test config property caching."""
        config_manager = ConfigManager("nonexistent_config.yaml")
        
        # First access should load config
        config1 = config_manager.config
        assert isinstance(config1, AppConfig)
        
        # Second access should return cached config
        config2 = config_manager.config
        assert config1 is config2


class TestSampleImageGeneration:
    """Test cases for sample image generation."""
    
    def test_create_sample_image(self):
        """Test sample image creation."""
        image = create_sample_image_with_faces()
        
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # Should be color image
        assert image.shape[2] == 3    # Should have 3 channels (BGR)
        assert image.dtype == np.uint8
    
    def test_sample_image_dimensions(self):
        """Test sample image dimensions."""
        image = create_sample_image_with_faces()
        
        # Should be 400x600x3 as defined in the function
        assert image.shape == (400, 600, 3)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_detection(self):
        """Test end-to-end detection workflow."""
        # Create sample image
        image = create_sample_image_with_faces()
        
        # Detect faces
        detector = FaceDetector(method="opencv")
        result = detector.detect_faces(image)
        
        # Draw detections
        output_image = detector.draw_detections(image, result)
        
        # Verify results
        assert isinstance(result, FaceDetectionResult)
        assert isinstance(output_image, np.ndarray)
        assert output_image.shape == image.shape
    
    def test_multiple_detection_methods(self):
        """Test multiple detection methods if available."""
        image = create_sample_image_with_faces()
        
        methods_to_test = ["opencv"]
        
        # Check for optional dependencies
        try:
            from mtcnn import MTCNN
            methods_to_test.append("mtcnn")
        except ImportError:
            pass
        
        try:
            import mediapipe as mp
            methods_to_test.append("mediapipe")
        except ImportError:
            pass
        
        try:
            import torch
            from facenet_pytorch import MTCNN as FacenetMTCNN
            methods_to_test.append("facenet")
        except ImportError:
            pass
        
        for method in methods_to_test:
            try:
                detector = FaceDetector(method=method)
                result = detector.detect_faces(image)
                assert isinstance(result, FaceDetectionResult)
                assert result.method == method
            except Exception as e:
                pytest.fail(f"Method {method} failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
