# Face Detection Implementation

A face detection project supporting multiple detection methods with both web interface and command-line tools.

## Features

- **Multiple Detection Methods**: OpenCV Haar Cascades, MTCNN, MediaPipe, and FaceNet
- **Web Interface**: Streamlit-based interactive demo
- **Command-Line Tools**: Batch processing and method comparison
- **Modern Architecture**: Type hints, configuration management, logging
- **Comprehensive Testing**: Unit tests and integration tests
- **Easy Setup**: Simple installation and configuration

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Face-Detection-Implementation.git
   cd Face-Detection-Implementation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web interface**:
   ```bash
   streamlit run web_app/app.py
   ```

4. **Or use the command line**:
   ```bash
   python -m src.cli sample --method opencv
   ```

## 📁 Project Structure

```
face-detection-implementation/
├── src/                          # Source code
│   ├── face_detector.py         # Core detection logic
│   ├── config_manager.py        # Configuration management
│   └── cli.py                   # Command-line interface
├── web_app/                     # Web interface
│   └── app.py                   # Streamlit application
├── tests/                       # Test suite
│   └── test_face_detection.py   # Unit and integration tests
├── config/                      # Configuration files
│   └── config.yaml             # Main configuration
├── data/                        # Sample data and outputs
├── models/                      # Model files (if any)
├── output/                      # Generated outputs
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 Detection Methods

### 1. OpenCV Haar Cascades (Default)
- **Pros**: Fast, lightweight, no additional dependencies
- **Cons**: Less accurate on challenging images
- **Best for**: Real-time applications, resource-constrained environments

### 2. MTCNN (Multi-task CNN)
- **Pros**: High accuracy, landmark detection
- **Cons**: Slower, requires TensorFlow
- **Install**: `pip install mtcnn`

### 3. MediaPipe
- **Pros**: Good balance of speed and accuracy, Google's solution
- **Cons**: Limited customization
- **Install**: `pip install mediapipe`

### 4. FaceNet PyTorch
- **Pros**: State-of-the-art accuracy
- **Cons**: Requires PyTorch, slower
- **Install**: `pip install torch torchvision facenet-pytorch`

## Web Interface

The Streamlit web interface provides:

- **Interactive Detection**: Upload images or use sample images
- **Method Comparison**: Compare different detection methods
- **Real-time Results**: See detection results with confidence scores
- **Export Options**: Download results and statistics
- **Configuration**: Adjust parameters through the sidebar

### Running the Web Interface

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

## Command Line Interface

### Single Image Detection

```bash
# Basic detection
python -m src.cli detect image.jpg --method opencv

# With custom confidence threshold
python -m src.cli detect image.jpg --method mtcnn --confidence 0.7

# Save result image
python -m src.cli detect image.jpg --method opencv --output result.jpg

# Output JSON results
python -m src.cli detect image.jpg --method opencv --json
```

### Batch Processing

```bash
# Process all images in a folder
python -m src.cli batch input_folder/ --method opencv --output_folder results/

# Process specific file types
python -m src.cli batch input_folder/ --extensions .jpg .png --method mtcnn
```

### Generate Sample Image

```bash
# Generate and test with sample image
python -m src.cli sample --method opencv --output sample_result.jpg
```

### Compare Methods

```bash
# Compare all available methods
python -m src.cli compare image.jpg --output comparison_results/
```

## Configuration

Configuration is managed through `config/config.yaml`:

```yaml
detection:
  method: "opencv"              # Detection method
  confidence_threshold: 0.5     # Confidence threshold
  min_face_size: 30            # Minimum face size
  scale_factor: 1.1            # Scale factor for OpenCV
  min_neighbors: 5             # Min neighbors for OpenCV

ui:
  title: "Face Detection Demo"
  theme: "light"
  sidebar_width: 300
  max_image_size: 800

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_face_detection.py
```

## Performance Comparison

| Method | Speed | Accuracy | Dependencies | Use Case |
|--------|-------|----------|--------------|----------|
| OpenCV | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | None | Real-time, embedded |
| MTCNN | ⭐⭐⭐ | ⭐⭐⭐⭐ | TensorFlow | High accuracy needed |
| MediaPipe | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | MediaPipe | Balanced performance |
| FaceNet | ⭐⭐ | ⭐⭐⭐⭐⭐ | PyTorch | Research, best accuracy |

## Usage Examples

### Python API

```python
from src.face_detector import FaceDetector, create_sample_image_with_faces
import cv2

# Create detector
detector = FaceDetector(method="opencv", confidence_threshold=0.5)

# Load image
image = cv2.imread("path/to/image.jpg")

# Detect faces
result = detector.detect_faces(image)

# Print results
print(f"Found {result.num_faces} faces")
for i, (x, y, w, h) in enumerate(result.bounding_boxes):
    print(f"Face {i+1}: x={x}, y={y}, w={w}, h={h}")

# Draw detections
output_image = detector.draw_detections(image, result)
cv2.imwrite("result.jpg", output_image)
```

### Configuration Management

```python
from src.config_manager import config_manager

# Load configuration
config = config_manager.config

# Access detection settings
print(f"Method: {config.detection.method}")
print(f"Confidence: {config.detection.confidence_threshold}")

# Modify and save configuration
config.detection.method = "mtcnn"
config_manager.save_config(config)
```

## 🛠️ Development

### Code Style

The project follows PEP 8 style guidelines. Use the provided tools:

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Detection Methods

1. Add the new method to `FaceDetector._initialize_detector()`
2. Implement the detection logic in a new `_detect_*` method
3. Add the method to the CLI and web interface options
4. Update tests and documentation

## Logging

The project includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Logs will show:
# - Detection method initialization
# - Processing progress
# - Error messages
# - Performance metrics
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Troubleshooting

### Common Issues

1. **ImportError for optional dependencies**:
   - Install the required packages: `pip install mtcnn mediapipe torch facenet-pytorch`

2. **OpenCV not found**:
   - Install OpenCV: `pip install opencv-python`

3. **CUDA/GPU issues**:
   - The project defaults to CPU usage for compatibility
   - Modify the detector initialization to use GPU if available

4. **Memory issues with large images**:
   - Reduce image size before processing
   - Use OpenCV method for better memory efficiency

### Getting Help

- Check the test files for usage examples
- Review the configuration options
- Run with verbose logging to debug issues

## Future Enhancements

- [ ] Real-time video processing
- [ ] Face recognition capabilities
- [ ] Emotion detection
- [ ] Age and gender estimation
- [ ] Docker containerization
- [ ] REST API interface
- [ ] Mobile app integration


# Face-Detection-Implementation
