#!/usr/bin/env python3
"""
Test script to verify the face detection implementation works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from face_detector import FaceDetector, create_sample_image_with_faces
from config_manager import config_manager
import cv2
import numpy as np


def test_basic_functionality():
    """Test basic face detection functionality."""
    print("🔍 Testing Face Detection Implementation")
    print("=" * 50)
    
    # Test configuration
    print("1. Testing configuration...")
    config = config_manager.config
    print(f"   ✓ Detection method: {config.detection.method}")
    print(f"   ✓ Confidence threshold: {config.detection.confidence_threshold}")
    
    # Test sample image generation
    print("\n2. Testing sample image generation...")
    sample_image = create_sample_image_with_faces()
    print(f"   ✓ Sample image created: {sample_image.shape}")
    
    # Save sample image
    cv2.imwrite("data/sample_image.jpg", sample_image)
    print("   ✓ Sample image saved to data/sample_image.jpg")
    
    # Test face detection
    print("\n3. Testing face detection...")
    detector = FaceDetector(method="opencv")
    result = detector.detect_faces(sample_image)
    
    print(f"   ✓ Detection method: {result.method}")
    print(f"   ✓ Faces detected: {result.num_faces}")
    print(f"   ✓ Bounding boxes: {len(result.bounding_boxes)}")
    
    # Test drawing detections
    print("\n4. Testing detection visualization...")
    output_image = detector.draw_detections(sample_image, result)
    cv2.imwrite("data/detection_result.jpg", output_image)
    print("   ✓ Detection result saved to data/detection_result.jpg")
    
    # Test file path input
    print("\n5. Testing file path input...")
    result_from_file = detector.detect_faces("data/sample_image.jpg")
    print(f"   ✓ File path detection: {result_from_file.num_faces} faces")
    
    print("\n✅ All basic tests passed!")
    return True


def test_advanced_features():
    """Test advanced features if dependencies are available."""
    print("\n🚀 Testing Advanced Features")
    print("=" * 50)
    
    sample_image = create_sample_image_with_faces()
    
    # Test MTCNN if available
    try:
        from mtcnn import MTCNN
        print("1. Testing MTCNN...")
        detector = FaceDetector(method="mtcnn")
        result = detector.detect_faces(sample_image)
        print(f"   ✓ MTCNN: {result.num_faces} faces detected")
        cv2.imwrite("data/mtcnn_result.jpg", detector.draw_detections(sample_image, result))
    except ImportError:
        print("1. MTCNN not available (install with: pip install mtcnn)")
    
    # Test MediaPipe if available
    try:
        import mediapipe as mp
        print("2. Testing MediaPipe...")
        detector = FaceDetector(method="mediapipe")
        result = detector.detect_faces(sample_image)
        print(f"   ✓ MediaPipe: {result.num_faces} faces detected")
        cv2.imwrite("data/mediapipe_result.jpg", detector.draw_detections(sample_image, result))
    except ImportError:
        print("2. MediaPipe not available (install with: pip install mediapipe)")
    
    # Test FaceNet if available
    try:
        import torch
        from facenet_pytorch import MTCNN as FacenetMTCNN
        print("3. Testing FaceNet...")
        detector = FaceDetector(method="facenet")
        result = detector.detect_faces(sample_image)
        print(f"   ✓ FaceNet: {result.num_faces} faces detected")
        cv2.imwrite("data/facenet_result.jpg", detector.draw_detections(sample_image, result))
    except ImportError:
        print("3. FaceNet not available (install with: pip install torch facenet-pytorch)")


def main():
    """Main test function."""
    try:
        # Test basic functionality
        test_basic_functionality()
        
        # Test advanced features
        test_advanced_features()
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Run the web interface: streamlit run web_app/app.py")
        print("2. Try the CLI: python -m src.cli sample --method opencv")
        print("3. Run tests: pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
