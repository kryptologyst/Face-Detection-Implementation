#!/usr/bin/env python3
"""
Setup script for Face Detection Implementation.

This script helps users set up the project environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing basic requirements"):
        return False
    
    # Ask about optional dependencies
    print("\n🔧 Optional Dependencies:")
    print("The following packages provide advanced face detection methods:")
    print("1. MTCNN - High accuracy detection with landmarks")
    print("2. MediaPipe - Google's balanced solution")
    print("3. FaceNet PyTorch - State-of-the-art accuracy")
    
    install_optional = input("\nWould you like to install optional dependencies? (y/n): ").lower().strip()
    
    if install_optional in ['y', 'yes']:
        optional_packages = [
            ("pip install mtcnn", "Installing MTCNN"),
            ("pip install mediapipe", "Installing MediaPipe"),
            ("pip install torch torchvision facenet-pytorch", "Installing FaceNet PyTorch")
        ]
        
        for command, description in optional_packages:
            run_command(command, description)
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    directories = ['data', 'models', 'output', 'config']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    return run_command("python -m pytest tests/ -v", "Running test suite")


def main():
    """Main setup function."""
    print("🚀 Face Detection Implementation Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but setup may still be functional")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run web_app/app.py")
    print("2. Try the CLI: python -m src.cli sample --method opencv")
    print("3. Check the README.md for more information")
    
    # Test basic functionality
    print("\n🔍 Testing basic functionality...")
    try:
        from src.face_detector import FaceDetector, create_sample_image_with_faces
        import cv2
        
        # Create sample image
        sample_image = create_sample_image_with_faces()
        
        # Test detection
        detector = FaceDetector(method="opencv")
        result = detector.detect_faces(sample_image)
        
        print(f"✅ Basic test passed: Detected {result.num_faces} faces")
        
        # Save test result
        output_image = detector.draw_detections(sample_image, result)
        cv2.imwrite("data/setup_test_result.jpg", output_image)
        print("✅ Test result saved to data/setup_test_result.jpg")
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
