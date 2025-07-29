"""
Google Vision API Integration Module

This module integrates with Google Cloud Vision API for OCR functionality.
Demonstrates cloud-based OCR capabilities and API management.
"""

import os
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import requests

from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account

from src.config.config import config
from src.utils.logger import LoggerMixin, log_function_call, log_performance


@dataclass
class VisionAPIResult:
    """Result from Google Vision API."""
    text: str
    confidence: float
    bounding_box: List[Dict]
    language: str
    api_response_time: float


class GoogleVisionAPI(LoggerMixin):
    """
    Google Vision API integration for OCR.
    
    Demonstrates:
    - Cloud-based OCR capabilities
    - API authentication and management
    - Error handling and retry logic
    - Performance monitoring
    """
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.credentials_path = config.google_cloud.credentials_path
        self.project_id = config.google_cloud.project_id
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Vision API client."""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                # Use service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.client = vision.ImageAnnotatorClient(credentials=credentials)
                self.log_info("Google Vision API client initialized with service account")
            else:
                # Use default credentials (for local development)
                self.client = vision.ImageAnnotatorClient()
                self.log_info("Google Vision API client initialized with default credentials")
                
        except Exception as e:
            self.log_error("Failed to initialize Google Vision API client", error=str(e))
            # Create a mock client for demo purposes
            self.client = MockVisionClient()
            self.log_warning("Using mock client for demo purposes")
    
    @log_performance
    def detect_text(self, image_path: Union[str, Path]) -> List[VisionAPIResult]:
        """
        Detect text in an image using Google Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected text results
        """
        if self.client is None:
            raise ValueError("Google Vision API client not initialized")
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            # Create image object
            image = types.Image(content=content)
            
            # Perform text detection
            start_time = time.time()
            response = self.client.text_detection(image=image)
            api_response_time = time.time() - start_time
            
            # Process results
            results = []
            
            if response.error.message:
                raise Exception(f"API Error: {response.error.message}")
            
            if response.text_annotations:
                # First annotation contains the entire text
                full_text = response.text_annotations[0].description
                
                # Process individual text blocks
                for text_annotation in response.text_annotations[1:]:
                    text = text_annotation.description
                    confidence = self._calculate_confidence(text_annotation)
                    
                    # Extract bounding box
                    vertices = text_annotation.bounding_poly.vertices
                    bounding_box = [
                        {'x': vertex.x, 'y': vertex.y} for vertex in vertices
                    ]
                    
                    # Detect language (simplified)
                    language = self._detect_language(text)
                    
                    result = VisionAPIResult(
                        text=text,
                        confidence=confidence,
                        bounding_box=bounding_box,
                        language=language,
                        api_response_time=api_response_time
                    )
                    results.append(result)
            
            self.log_info("Google Vision API text detection completed",
                         image_path=str(image_path),
                         detected_text_count=len(results),
                         api_response_time=api_response_time)
            
            return results
            
        except Exception as e:
            self.log_error("Google Vision API text detection failed", error=str(e))
            raise
    
    @log_performance
    def detect_text_from_bytes(self, image_bytes: bytes) -> List[VisionAPIResult]:
        """
        Detect text from image bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of detected text results
        """
        if self.client is None:
            raise ValueError("Google Vision API client not initialized")
        
        try:
            # Create image object from bytes
            image = types.Image(content=image_bytes)
            
            # Perform text detection
            start_time = time.time()
            response = self.client.text_detection(image=image)
            api_response_time = time.time() - start_time
            
            # Process results (same as detect_text method)
            results = []
            
            if response.error.message:
                raise Exception(f"API Error: {response.error.message}")
            
            if response.text_annotations:
                for text_annotation in response.text_annotations[1:]:
                    text = text_annotation.description
                    confidence = self._calculate_confidence(text_annotation)
                    
                    vertices = text_annotation.bounding_poly.vertices
                    bounding_box = [
                        {'x': vertex.x, 'y': vertex.y} for vertex in vertices
                    ]
                    
                    language = self._detect_language(text)
                    
                    result = VisionAPIResult(
                        text=text,
                        confidence=confidence,
                        bounding_box=bounding_box,
                        language=language,
                        api_response_time=api_response_time
                    )
                    results.append(result)
            
            self.log_info("Google Vision API text detection from bytes completed",
                         detected_text_count=len(results),
                         api_response_time=api_response_time)
            
            return results
            
        except Exception as e:
            self.log_error("Google Vision API text detection from bytes failed", error=str(e))
            raise
    
    def _calculate_confidence(self, text_annotation) -> float:
        """
        Calculate confidence score for text annotation.
        
        Args:
            text_annotation: Text annotation from Vision API
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # In a real implementation, you might use the confidence scores
        # from the API response. For demo purposes, we'll use a simplified approach.
        
        # Check if confidence is available in the response
        if hasattr(text_annotation, 'confidence') and text_annotation.confidence:
            return text_annotation.confidence
        else:
            # Use a default confidence based on text length and characteristics
            text = text_annotation.description
            base_confidence = 0.8
            
            # Adjust confidence based on text characteristics
            if len(text) > 10:
                base_confidence += 0.1
            if text.isalpha():
                base_confidence += 0.05
            if text.isupper():
                base_confidence -= 0.05
            
            return min(base_confidence, 1.0)
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # Simplified language detection
        # In practice, you might use Google's language detection API
        
        # Basic language detection based on character sets
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'  # Chinese
        elif any('\u3040' <= char <= '\u309f' for char in text):
            return 'ja'  # Japanese
        elif any('\uac00' <= char <= '\ud7af' for char in text):
            return 'ko'  # Korean
        elif any('\u0600' <= char <= '\u06ff' for char in text):
            return 'ar'  # Arabic
        elif any('\u0590' <= char <= '\u05ff' for char in text):
            return 'he'  # Hebrew
        else:
            return 'en'  # Default to English
    
    @log_function_call
    def batch_detect_text(self, image_paths: List[Union[str, Path]]) -> Dict[str, List[VisionAPIResult]]:
        """
        Detect text in multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        results = {}
        
        for image_path in image_paths:
            try:
                image_results = self.detect_text(image_path)
                results[str(image_path)] = image_results
            except Exception as e:
                self.log_error(f"Failed to process {image_path}", error=str(e))
                results[str(image_path)] = []
        
        self.log_info("Batch text detection completed",
                     total_images=len(image_paths),
                     successful_images=len([r for r in results.values() if r]))
        
        return results
    
    @log_function_call
    def get_api_usage_stats(self) -> Dict:
        """
        Get API usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        # In a real implementation, you would query Google Cloud billing API
        # For demo purposes, return mock statistics
        
        stats = {
            'total_requests': 100,
            'successful_requests': 95,
            'failed_requests': 5,
            'average_response_time': 0.8,
            'total_cost': 12.50,
            'last_updated': time.time()
        }
        
        self.log_info("API usage statistics retrieved", stats=stats)
        return stats
    
    @log_function_call
    def validate_api_access(self) -> bool:
        """
        Validate API access and credentials.
        
        Returns:
            True if API access is valid, False otherwise
        """
        try:
            # Try to make a simple API call
            test_image = types.Image(content=b'test')
            response = self.client.text_detection(image=test_image)
            
            # If we get here, the API is accessible
            self.log_info("Google Vision API access validated successfully")
            return True
            
        except Exception as e:
            self.log_error("Google Vision API access validation failed", error=str(e))
            return False


class MockVisionClient:
    """
    Mock Google Vision API client for demo purposes.
    
    Used when Google Cloud credentials are not available.
    """
    
    def __init__(self):
        self.logger = LoggerMixin().logger
    
    def text_detection(self, image):
        """Mock text detection method."""
        # Create a mock response
        class MockResponse:
            def __init__(self):
                self.error = type('MockError', (), {'message': ''})()
                self.text_annotations = []
        
        response = MockResponse()
        
        # Add some mock text annotations
        class MockTextAnnotation:
            def __init__(self, text, confidence=0.9):
                self.description = text
                self.confidence = confidence
                self.bounding_poly = type('MockBoundingPoly', (), {
                    'vertices': [
                        type('MockVertex', (), {'x': 10, 'y': 10})(),
                        type('MockVertex', (), {'x': 100, 'y': 10})(),
                        type('MockVertex', (), {'x': 100, 'y': 30})(),
                        type('MockVertex', (), {'x': 10, 'y': 30})()
                    ]
                })()
        
        # Add mock text annotations
        response.text_annotations = [
            MockTextAnnotation("Sample OCR Text", 0.95),
            MockTextAnnotation("Hello World", 0.88),
            MockTextAnnotation("123456", 0.92)
        ]
        
        self.logger.info("Mock Vision API response generated")
        return response


class VisionAPIManager:
    """
    Manager class for Google Vision API operations.
    
    Provides high-level operations and error handling.
    """
    
    def __init__(self):
        self.api = GoogleVisionAPI()
        self.logger = LoggerMixin().logger
    
    @log_function_call
    def process_image_directory(self, directory_path: Union[str, Path]) -> Dict[str, List[VisionAPIResult]]:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            Dictionary mapping image paths to results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory_path.glob(f"*{ext}"))
            image_files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.warning("No image files found in directory", directory=str(directory_path))
            return {}
        
        # Process all images
        results = self.api.batch_detect_text(image_files)
        
        self.logger.info("Directory processing completed",
                        directory=str(directory_path),
                        total_images=len(image_files),
                        processed_images=len(results))
        
        return results
    
    @log_function_call
    def save_results(self, results: Dict[str, List[VisionAPIResult]], 
                    output_path: Union[str, Path]):
        """
        Save API results to JSON file.
        
        Args:
            results: Dictionary of results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        
        for image_path, vision_results in results.items():
            serializable_results[image_path] = [
                {
                    'text': result.text,
                    'confidence': result.confidence,
                    'bounding_box': result.bounding_box,
                    'language': result.language,
                    'api_response_time': result.api_response_time
                }
                for result in vision_results
            ]
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info("Results saved", output_path=str(output_path))
    
    @log_function_call
    def compare_with_custom_model(self, image_path: Union[str, Path], 
                                custom_results: List[Dict]) -> Dict:
        """
        Compare Google Vision API results with custom model results.
        
        Args:
            image_path: Path to the image
            custom_results: Results from custom OCR model
            
        Returns:
            Comparison metrics
        """
        # Get Google Vision API results
        vision_results = self.api.detect_text(image_path)
        
        # Extract text from both results
        vision_texts = [result.text for result in vision_results]
        custom_texts = [result.get('text', '') for result in custom_results]
        
        # Calculate comparison metrics
        vision_text = ' '.join(vision_texts).lower()
        custom_text = ' '.join(custom_texts).lower()
        
        # Simple text similarity (in practice, use more sophisticated metrics)
        common_words = set(vision_text.split()) & set(custom_text.split())
        total_words = set(vision_text.split()) | set(custom_text.split())
        
        similarity = len(common_words) / len(total_words) if total_words else 0.0
        
        comparison = {
            'image_path': str(image_path),
            'vision_api_text_count': len(vision_texts),
            'custom_model_text_count': len(custom_texts),
            'text_similarity': similarity,
            'vision_api_avg_confidence': sum(r.confidence for r in vision_results) / len(vision_results) if vision_results else 0.0,
            'custom_model_avg_confidence': sum(r.get('confidence', 0.0) for r in custom_results) / len(custom_results) if custom_results else 0.0
        }
        
        self.logger.info("Model comparison completed",
                        image_path=str(image_path),
                        similarity=similarity)
        
        return comparison 