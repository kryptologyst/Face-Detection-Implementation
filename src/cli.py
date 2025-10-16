"""
Command-line interface for face detection.

This module provides a CLI for batch processing images and testing
different detection methods from the command line.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json
import time

import cv2
import numpy as np

# Import our custom modules
from src.face_detector import FaceDetector, FaceDetectionResult, create_sample_image_with_faces
from src.config_manager import config_manager


class FaceDetectionCLI:
    """Command-line interface for face detection."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.config = config_manager.config
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format
        )
        self.logger = logging.getLogger(__name__)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Face Detection CLI - Detect faces in images using various methods",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Detect faces in a single image
  python -m src.cli detect image.jpg --method opencv --output result.jpg
  
  # Batch process multiple images
  python -m src.cli batch input_folder/ --method mtcnn --output_folder results/
  
  # Generate sample image and test detection
  python -m src.cli sample --method mediapipe
  
  # Compare all available methods
  python -m src.cli compare image.jpg
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Detect command
        detect_parser = subparsers.add_parser('detect', help='Detect faces in a single image')
        detect_parser.add_argument('image', help='Path to input image')
        detect_parser.add_argument('--method', '-m', 
                                  choices=['opencv', 'mtcnn', 'mediapipe', 'facenet'],
                                  default='opencv',
                                  help='Detection method to use')
        detect_parser.add_argument('--output', '-o', help='Output image path')
        detect_parser.add_argument('--confidence', '-c', type=float, default=0.5,
                                  help='Confidence threshold (0.0-1.0)')
        detect_parser.add_argument('--json', action='store_true',
                                  help='Output results as JSON')
        
        # Batch command
        batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
        batch_parser.add_argument('input_folder', help='Path to input folder')
        batch_parser.add_argument('--method', '-m',
                                 choices=['opencv', 'mtcnn', 'mediapipe', 'facenet'],
                                 default='opencv',
                                 help='Detection method to use')
        batch_parser.add_argument('--output_folder', '-o', help='Output folder path')
        batch_parser.add_argument('--confidence', '-c', type=float, default=0.5,
                                 help='Confidence threshold (0.0-1.0)')
        batch_parser.add_argument('--extensions', nargs='+', 
                                 default=['.jpg', '.jpeg', '.png', '.bmp'],
                                 help='Image file extensions to process')
        
        # Sample command
        sample_parser = subparsers.add_parser('sample', help='Generate and test with sample image')
        sample_parser.add_argument('--method', '-m',
                                  choices=['opencv', 'mtcnn', 'mediapipe', 'facenet'],
                                  default='opencv',
                                  help='Detection method to use')
        sample_parser.add_argument('--output', '-o', default='sample_result.jpg',
                                  help='Output image path')
        
        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare all available methods')
        compare_parser.add_argument('image', help='Path to input image')
        compare_parser.add_argument('--output', '-o', help='Output folder for results')
        
        return parser
    
    def detect_single_image(self, args) -> None:
        """Detect faces in a single image."""
        image_path = Path(args.image)
        
        if not image_path.exists():
            self.logger.error(f"Image file not found: {image_path}")
            sys.exit(1)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            sys.exit(1)
        
        # Detect faces
        self.logger.info(f"Detecting faces using {args.method} method...")
        detector = FaceDetector(method=args.method, confidence_threshold=args.confidence)
        
        start_time = time.time()
        result = detector.detect_faces(image)
        detection_time = time.time() - start_time
        
        # Print results
        print(f"Detection completed in {detection_time:.3f} seconds")
        print(f"Found {result.num_faces} faces")
        
        if result.confidence_scores:
            print("Confidence scores:")
            for i, score in enumerate(result.confidence_scores):
                print(f"  Face {i+1}: {score:.3f}")
        
        # Output JSON if requested
        if args.json:
            output_data = {
                "method": args.method,
                "num_faces": result.num_faces,
                "detection_time": detection_time,
                "confidence_threshold": args.confidence,
                "bounding_boxes": result.bounding_boxes,
                "confidence_scores": result.confidence_scores
            }
            print(json.dumps(output_data, indent=2))
        
        # Save output image if specified
        if args.output:
            output_image = detector.draw_detections(image, result)
            cv2.imwrite(args.output, output_image)
            print(f"Result saved to: {args.output}")
    
    def batch_process(self, args) -> None:
        """Batch process multiple images."""
        input_folder = Path(args.input_folder)
        output_folder = Path(args.output_folder) if args.output_folder else None
        
        if not input_folder.exists():
            self.logger.error(f"Input folder not found: {input_folder}")
            sys.exit(1)
        
        # Find all image files
        image_files = []
        for ext in args.extensions:
            image_files.extend(input_folder.glob(f"*{ext}"))
            image_files.extend(input_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.error(f"No image files found in {input_folder}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images to process")
        
        # Initialize detector
        detector = FaceDetector(method=args.method, confidence_threshold=args.confidence)
        
        # Process each image
        results = []
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                image = cv2.imread(str(image_file))
                if image is None:
                    self.logger.warning(f"Could not load image: {image_file}")
                    continue
                
                start_time = time.time()
                result = detector.detect_faces(image)
                detection_time = time.time() - start_time
                
                # Save result image if output folder specified
                if output_folder:
                    output_folder.mkdir(parents=True, exist_ok=True)
                    output_image = detector.draw_detections(image, result)
                    output_path = output_folder / f"result_{image_file.name}"
                    cv2.imwrite(str(output_path), output_image)
                
                results.append({
                    "file": image_file.name,
                    "faces": result.num_faces,
                    "time": detection_time,
                    "confidence_scores": result.confidence_scores
                })
                
                print(f"  Found {result.num_faces} faces in {detection_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")
                continue
        
        # Print summary
        total_faces = sum(r["faces"] for r in results)
        total_time = sum(r["time"] for r in results)
        print(f"\nBatch processing completed:")
        print(f"  Total images: {len(results)}")
        print(f"  Total faces: {total_faces}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per image: {total_time/len(results):.3f}s")
    
    def generate_sample(self, args) -> None:
        """Generate sample image and test detection."""
        print("Generating sample image with faces...")
        sample_image = create_sample_image_with_faces()
        
        # Save sample image
        sample_path = "sample_image.jpg"
        cv2.imwrite(sample_path, sample_image)
        print(f"Sample image saved to: {sample_path}")
        
        # Detect faces
        print(f"Detecting faces using {args.method} method...")
        detector = FaceDetector(method=args.method)
        
        start_time = time.time()
        result = detector.detect_faces(sample_image)
        detection_time = time.time() - start_time
        
        print(f"Detection completed in {detection_time:.3f} seconds")
        print(f"Found {result.num_faces} faces")
        
        # Save result
        output_image = detector.draw_detections(sample_image, result)
        cv2.imwrite(args.output, output_image)
        print(f"Result saved to: {args.output}")
    
    def compare_methods(self, args) -> None:
        """Compare all available detection methods."""
        image_path = Path(args.image)
        
        if not image_path.exists():
            self.logger.error(f"Image file not found: {image_path}")
            sys.exit(1)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            sys.exit(1)
        
        # Available methods
        available_methods = ["opencv"]
        
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
        
        print(f"Comparing {len(available_methods)} methods: {', '.join(available_methods)}")
        print("-" * 60)
        
        results = []
        output_folder = Path(args.output) if args.output else None
        
        for method in available_methods:
            try:
                print(f"Testing {method.upper()}...")
                detector = FaceDetector(method=method)
                
                start_time = time.time()
                result = detector.detect_faces(image)
                detection_time = time.time() - start_time
                
                results.append({
                    "method": method,
                    "faces": result.num_faces,
                    "time": detection_time,
                    "confidence_scores": result.confidence_scores
                })
                
                print(f"  Faces: {result.num_faces}, Time: {detection_time:.3f}s")
                
                # Save result image if output folder specified
                if output_folder:
                    output_folder.mkdir(parents=True, exist_ok=True)
                    output_image = detector.draw_detections(image, result)
                    output_path = output_folder / f"result_{method}.jpg"
                    cv2.imwrite(str(output_path), output_image)
                
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "method": method,
                    "faces": "Error",
                    "time": "N/A",
                    "confidence_scores": []
                })
        
        # Print comparison table
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"{'Method':<12} {'Faces':<8} {'Time (s)':<10} {'Avg Confidence':<15}")
        print("-" * 60)
        
        for result in results:
            avg_conf = "N/A"
            if result["confidence_scores"]:
                avg_conf = f"{np.mean(result['confidence_scores']):.3f}"
            
            print(f"{result['method'].upper():<12} {result['faces']:<8} {result['time']:<10} {avg_conf:<15}")
    
    def run(self) -> None:
        """Run the CLI."""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
        
        try:
            if args.command == 'detect':
                self.detect_single_image(args)
            elif args.command == 'batch':
                self.batch_process(args)
            elif args.command == 'sample':
                self.generate_sample(args)
            elif args.command == 'compare':
                self.compare_methods(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli = FaceDetectionCLI()
    cli.run()


if __name__ == "__main__":
    main()
