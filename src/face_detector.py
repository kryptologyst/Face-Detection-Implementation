"""
Modern Face Detection Implementation

This module provides multiple face detection methods including:
- OpenCV Haar Cascades (classic, fast)
- MTCNN (Multi-task CNN, more accurate)
- RetinaFace (state-of-the-art accuracy)
- MediaPipe (Google's solution, good balance)

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from pathlib import Path

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logging.warning("MTCNN not available. Install with: pip install mtcnn")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Install with: pip install mediapipe")

try:
    import torch
    import torchvision.transforms as transforms
    from facenet_pytorch import MTCNN as FacenetMTCNN
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    logging.warning("FaceNet PyTorch not available. Install with: pip install facenet-pytorch")


class FaceDetectionResult:
    """Container for face detection results."""
    
    def __init__(self, 
                 bounding_boxes: List[Tuple[int, int, int, int]], 
                 confidence_scores: Optional[List[float]] = None,
                 landmarks: Optional[List[List[Tuple[int, int]]]] = None,
                 method: str = "unknown"):
        self.bounding_boxes = bounding_boxes
        self.confidence_scores = confidence_scores or []
        self.landmarks = landmarks or []
        self.method = method
        self.num_faces = len(bounding_boxes)
    
    def __repr__(self) -> str:
        return f"FaceDetectionResult(method={self.method}, faces={self.num_faces})"


class FaceDetector:
    """Modern face detector supporting multiple detection methods."""
    
    def __init__(self, method: str = "opencv", confidence_threshold: float = 0.5):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('opencv', 'mtcnn', 'mediapipe', 'facenet')
            confidence_threshold: Minimum confidence for detections
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize detector based on method
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize the selected detection method."""
        if self.method == "opencv":
            self._init_opencv()
        elif self.method == "mtcnn":
            self._init_mtcnn()
        elif self.method == "mediapipe":
            self._init_mediapipe()
        elif self.method == "facenet":
            self._init_facenet()
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _init_opencv(self) -> None:
        """Initialize OpenCV Haar Cascade detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        self.logger.info("OpenCV Haar Cascade detector initialized")
    
    def _init_mtcnn(self) -> None:
        """Initialize MTCNN detector."""
        if not MTCNN_AVAILABLE:
            raise ImportError("MTCNN not available. Install with: pip install mtcnn")
        self.detector = MTCNN()
        self.logger.info("MTCNN detector initialized")
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe face detection."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for close-range, 1 for full-range
            min_detection_confidence=self.confidence_threshold
        )
        self.logger.info("MediaPipe face detector initialized")
    
    def _init_facenet(self) -> None:
        """Initialize FaceNet MTCNN detector."""
        if not FACENET_AVAILABLE:
            raise ImportError("FaceNet PyTorch not available. Install with: pip install facenet-pytorch")
        self.detector = FacenetMTCNN(
            keep_all=True,
            device='cpu'  # Use CPU by default, can be changed to 'cuda' if available
        )
        self.logger.info("FaceNet MTCNN detector initialized")
    
    def detect_faces(self, image: Union[np.ndarray, str, Path]) -> FaceDetectionResult:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array, file path, or Path object
            
        Returns:
            FaceDetectionResult containing bounding boxes and metadata
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Convert to RGB if needed (for some detectors)
        if self.method in ["mtcnn", "mediapipe", "facenet"]:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Detect faces using selected method
        if self.method == "opencv":
            return self._detect_opencv(image)
        elif self.method == "mtcnn":
            return self._detect_mtcnn(image_rgb)
        elif self.method == "mediapipe":
            return self._detect_mediapipe(image_rgb)
        elif self.method == "facenet":
            return self._detect_facenet(image_rgb)
    
    def _detect_opencv(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect faces using OpenCV Haar Cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        bounding_boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        return FaceDetectionResult(bounding_boxes, method="opencv")
    
    def _detect_mtcnn(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect faces using MTCNN."""
        results = self.detector.detect_faces(image)
        
        bounding_boxes = []
        confidence_scores = []
        landmarks = []
        
        for result in results:
            if result['confidence'] >= self.confidence_threshold:
                x, y, w, h = result['box']
                bounding_boxes.append((int(x), int(y), int(w), int(h)))
                confidence_scores.append(result['confidence'])
                landmarks.append(result['keypoints'])
        
        return FaceDetectionResult(
            bounding_boxes, 
            confidence_scores, 
            landmarks, 
            method="mtcnn"
        )
    
    def _detect_mediapipe(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect faces using MediaPipe."""
        results = self.detector.process(image)
        
        bounding_boxes = []
        confidence_scores = []
        
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                bounding_boxes.append((x, y, width, height))
                confidence_scores.append(detection.score[0])
        
        return FaceDetectionResult(
            bounding_boxes, 
            confidence_scores, 
            method="mediapipe"
        )
    
    def _detect_facenet(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect faces using FaceNet MTCNN."""
        # Convert numpy array to PIL Image
        from PIL import Image
        pil_image = Image.fromarray(image)
        
        # Detect faces
        boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
        
        bounding_boxes = []
        confidence_scores = []
        landmark_list = []
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    bounding_boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
                    confidence_scores.append(float(prob))
                    
                    if landmarks is not None:
                        landmark_list.append([(int(x), int(y)) for x, y in landmarks[i]])
        
        return FaceDetectionResult(
            bounding_boxes, 
            confidence_scores, 
            landmark_list, 
            method="facenet"
        )
    
    def draw_detections(self, image: np.ndarray, result: FaceDetectionResult) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on the image.
        
        Args:
            image: Input image
            result: Detection results
            
        Returns:
            Image with drawn detections
        """
        output_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(result.bounding_boxes):
            # Draw bounding box
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence score if available
            if i < len(result.confidence_scores):
                confidence = result.confidence_scores[i]
                cv2.putText(output_image, f"{confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw landmarks if available
            if i < len(result.landmarks) and result.landmarks[i]:
                for landmark in result.landmarks[i]:
                    cv2.circle(output_image, landmark, 2, (255, 0, 0), -1)
        
        return output_image


def create_sample_image_with_faces() -> np.ndarray:
    """Create a synthetic image with face-like shapes for testing."""
    # Create a simple synthetic image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw simple face-like shapes
    faces = [
        (100, 100, 80, 80),   # Face 1
        (300, 150, 90, 90),   # Face 2
        (500, 200, 70, 70),   # Face 3
    ]
    
    for x, y, w, h in faces:
        # Draw face outline
        cv2.ellipse(image, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (200, 200, 200), -1)
        cv2.ellipse(image, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (0, 0, 0), 2)
        
        # Draw eyes
        cv2.circle(image, (x+w//3, y+h//3), 5, (0, 0, 0), -1)
        cv2.circle(image, (x+2*w//3, y+h//3), 5, (0, 0, 0), -1)
        
        # Draw nose
        cv2.circle(image, (x+w//2, y+h//2), 3, (0, 0, 0), -1)
        
        # Draw mouth
        cv2.ellipse(image, (x+w//2, y+2*h//3), (w//6, h//12), 0, 0, 180, (0, 0, 0), 2)
    
    return image


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample image
    sample_image = create_sample_image_with_faces()
    
    # Test different detection methods
    methods = ["opencv"]
    if MTCNN_AVAILABLE:
        methods.append("mtcnn")
    if MEDIAPIPE_AVAILABLE:
        methods.append("mediapipe")
    if FACENET_AVAILABLE:
        methods.append("facenet")
    
    for method in methods:
        try:
            detector = FaceDetector(method=method)
            result = detector.detect_faces(sample_image)
            print(f"{method.upper()}: Detected {result.num_faces} faces")
            
            # Draw and save result
            output_image = detector.draw_detections(sample_image, result)
            cv2.imwrite(f"data/detection_result_{method}.jpg", output_image)
            
        except Exception as e:
            print(f"Error with {method}: {e}")
