"""
Image preprocessing utilities for the IDP System.

This module provides image preprocessing functionality specifically designed
for improving OCR accuracy on document images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import logging

# Setup logger
logger = logging.getLogger(__name__)


class PreprocessingMethod(Enum):
    """Enumeration of available preprocessing methods."""
    GRAYSCALE_CONVERSION = "grayscale_conversion"
    NOISE_REDUCTION = "noise_reduction"
    CONTRAST_ENHANCEMENT = "contrast_enhancement" 
    BRIGHTNESS_ADJUSTMENT = "brightness_adjustment"
    DESKEWING = "deskewing"
    BINARIZATION = "binarization"
    MORPHOLOGICAL_OPERATIONS = "morphological_operations"
    RESIZE_NORMALIZATION = "resize_normalization"
    SHADOW_REMOVAL = "shadow_removal"
    PERSPECTIVE_CORRECTION = "perspective_correction"


class ImagePreprocessor:
    """
    Comprehensive image preprocessing class for document images.
    
    This class provides various preprocessing methods to improve OCR accuracy
    on different types of document images.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.preprocessing_chain = []
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Enhance image contrast and brightness.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Enhanced image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def reduce_noise(image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Reduce noise in the image.
        
        Args:
            image: Input image
            method: Noise reduction method ('gaussian', 'median', 'bilateral')
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            logger.warning(f"Unknown noise reduction method: {method}. Using gaussian.")
            return cv2.GaussianBlur(image, (5, 5), 0)
    
    @staticmethod
    def apply_threshold(
        image: np.ndarray, 
        threshold_value: int = 127,
        max_value: int = 255,
        threshold_type: int = cv2.THRESH_BINARY
    ) -> np.ndarray:
        """
        Apply binary threshold to image.
        
        Args:
            image: Input image
            threshold_value: Threshold value
            max_value: Maximum value to use with threshold type
            threshold_type: Type of threshold
            
        Returns:
            Thresholded image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, thresholded = cv2.threshold(image, threshold_value, max_value, threshold_type)
        return thresholded
    
    @staticmethod
    def adaptive_threshold(
        image: np.ndarray, 
        max_value: int = 255,
        adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type: int = cv2.THRESH_BINARY,
        block_size: int = 11,
        c: int = 2
    ) -> np.ndarray:
        """
        Apply adaptive threshold to image.
        
        Args:
            image: Input image
            max_value: Maximum value to use
            adaptive_method: Adaptive method
            threshold_type: Type of threshold
            block_size: Size of the neighborhood area
            c: Constant subtracted from the mean
            
        Returns:
            Thresholded image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return cv2.adaptiveThreshold(
            image, max_value, adaptive_method, threshold_type, block_size, c
        )
    
    @staticmethod
    def resize(
        image: np.ndarray, 
        width: int, 
        height: int, 
        interpolation: int = cv2.INTER_AREA
    ) -> np.ndarray:
        """
        Resize image to specified dimensions.
        
        Args:
            image: Input image
            width: Target width
            height: Target height
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (width, height), interpolation=interpolation)
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to 0-255 range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            return image_norm.astype(np.uint8)
        return image
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Deskew image by detecting and correcting rotation.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough Transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate the most common angle
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    angles.append(angle)
                
                # Use median angle to avoid outliers
                skew_angle = np.median(angles)
                
                # Only correct if angle is significant
                if abs(skew_angle) > 0.5:
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    return cv2.warpAffine(image, rotation_matrix, (width, height))
            
            return image
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            return image
    
    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from document image.
        
        Args:
            image: Input image
            
        Returns:
            Image with shadows removed
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply morphological opening to detect shadows
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Create background model
            background = cv2.dilate(opened, kernel, iterations=1)
            
            # Normalize by background
            normalized = cv2.divide(gray, background, scale=255.0)
            
            return normalized.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Shadow removal failed: {str(e)}")
            return image
    
    @staticmethod
    def morphological_operations(
        image: np.ndarray, 
        operation: int = cv2.MORPH_CLOSE,
        kernel_size: Tuple[int, int] = (3, 3),
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological operations to image.
        
        Args:
            image: Input image
            operation: Morphological operation type
            kernel_size: Size of the morphological kernel
            iterations: Number of times to apply the operation
            
        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, operation, kernel, iterations=iterations)
    
    @staticmethod
    def correct_perspective(image: np.ndarray, corners: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Correct perspective distortion in document image.
        
        Args:
            image: Input image
            corners: Four corner points of the document (optional)
            
        Returns:
            Perspective-corrected image
        """
        try:
            if corners is None:
                # Auto-detect document corners
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                
                # Find contours
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Find the largest contour (assuming it's the document)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Approximate contour to get corner points
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    if len(approx) == 4:
                        corners = [tuple(point[0]) for point in approx]
                    else:
                        return image  # Cannot find proper corners
                else:
                    return image  # No contours found
            
            if len(corners) == 4:
                # Order corners: top-left, top-right, bottom-right, bottom-left
                corners = sorted(corners)
                tl, bl = sorted(corners[:2], key=lambda x: x[1])
                tr, br = sorted(corners[2:], key=lambda x: x[1])
                corners = [tl, tr, br, bl]
                
                # Calculate dimensions for the corrected image
                width = max(
                    int(np.linalg.norm(np.array(tr) - np.array(tl))),
                    int(np.linalg.norm(np.array(br) - np.array(bl)))
                )
                height = max(
                    int(np.linalg.norm(np.array(bl) - np.array(tl))),
                    int(np.linalg.norm(np.array(br) - np.array(tr)))
                )
                
                # Define destination points
                dst_corners = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                
                # Get perspective transform matrix
                src_corners = np.array(corners, dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
                
                # Apply perspective transformation
                corrected = cv2.warpPerspective(image, matrix, (width, height))
                return corrected
            
            return image
            
        except Exception as e:
            logger.warning(f"Perspective correction failed: {str(e)}")
            return image
    
    @staticmethod
    def sharpen(image: np.ndarray, kernel_type: str = 'unsharp') -> np.ndarray:
        """
        Sharpen image to improve text clarity.
        
        Args:
            image: Input image
            kernel_type: Type of sharpening kernel ('unsharp', 'laplacian')
            
        Returns:
            Sharpened image
        """
        try:
            if kernel_type == 'unsharp':
                # Create unsharp mask
                gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
                sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            elif kernel_type == 'laplacian':
                # Apply Laplacian sharpening
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharpened = cv2.filter2D(image, -1, kernel)
            else:
                logger.warning(f"Unknown sharpening kernel: {kernel_type}")
                return image
            
            return np.clip(sharpened, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Sharpening failed: {str(e)}")
            return image
    
    def add_preprocessing_step(self, method: PreprocessingMethod, **kwargs):
        """
        Add a preprocessing step to the chain.
        
        Args:
            method: Preprocessing method to add
            **kwargs: Parameters for the method
        """
        self.preprocessing_chain.append((method, kwargs))
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using the configured preprocessing chain.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        processed_image = image.copy()
        
        for method, kwargs in self.preprocessing_chain:
            try:
                if method == PreprocessingMethod.GRAYSCALE_CONVERSION:
                    processed_image = self.to_grayscale(processed_image)
                elif method == PreprocessingMethod.NOISE_REDUCTION:
                    processed_image = self.reduce_noise(processed_image, **kwargs)
                elif method == PreprocessingMethod.CONTRAST_ENHANCEMENT:
                    processed_image = self.enhance_contrast(processed_image, **kwargs)
                elif method == PreprocessingMethod.BINARIZATION:
                    if kwargs.get('adaptive', False):
                        processed_image = self.adaptive_threshold(processed_image, **kwargs)
                    else:
                        processed_image = self.apply_threshold(processed_image, **kwargs)
                elif method == PreprocessingMethod.DESKEWING:
                    processed_image = self.deskew(processed_image)
                elif method == PreprocessingMethod.SHADOW_REMOVAL:
                    processed_image = self.remove_shadows(processed_image)
                elif method == PreprocessingMethod.RESIZE_NORMALIZATION:
                    if 'width' in kwargs and 'height' in kwargs:
                        processed_image = self.resize(processed_image, **kwargs)
                    processed_image = self.normalize(processed_image)
                elif method == PreprocessingMethod.MORPHOLOGICAL_OPERATIONS:
                    processed_image = self.morphological_operations(processed_image, **kwargs)
                elif method == PreprocessingMethod.PERSPECTIVE_CORRECTION:
                    processed_image = self.correct_perspective(processed_image, **kwargs)
                    
            except Exception as e:
                logger.error(f"Error applying {method}: {str(e)}")
                continue
        
        return processed_image
    
    def reset_chain(self):
        """Reset the preprocessing chain."""
        self.preprocessing_chain = []
    
    @classmethod
    def create_document_pipeline(cls, document_type: str = 'general') -> 'ImagePreprocessor':
        """
        Create a preprocessor with a predefined pipeline for specific document types.
        
        Args:
            document_type: Type of document ('general', 'id_card', 'passport', 'receipt')
            
        Returns:
            Configured preprocessor instance
        """
        preprocessor = cls()
        
        if document_type == 'general':
            preprocessor.add_preprocessing_step(PreprocessingMethod.GRAYSCALE_CONVERSION)
            preprocessor.add_preprocessing_step(PreprocessingMethod.NOISE_REDUCTION, method='gaussian')
            preprocessor.add_preprocessing_step(PreprocessingMethod.CONTRAST_ENHANCEMENT, alpha=1.2)
            preprocessor.add_preprocessing_step(PreprocessingMethod.BINARIZATION, adaptive=True)
            
        elif document_type == 'id_card':
            preprocessor.add_preprocessing_step(PreprocessingMethod.PERSPECTIVE_CORRECTION)
            preprocessor.add_preprocessing_step(PreprocessingMethod.GRAYSCALE_CONVERSION)
            preprocessor.add_preprocessing_step(PreprocessingMethod.SHADOW_REMOVAL)
            preprocessor.add_preprocessing_step(PreprocessingMethod.CONTRAST_ENHANCEMENT, alpha=1.3)
            preprocessor.add_preprocessing_step(PreprocessingMethod.NOISE_REDUCTION, method='bilateral')
            preprocessor.add_preprocessing_step(PreprocessingMethod.BINARIZATION, adaptive=True, block_size=15)
            
        elif document_type == 'passport':
            preprocessor.add_preprocessing_step(PreprocessingMethod.DESKEWING)
            preprocessor.add_preprocessing_step(PreprocessingMethod.GRAYSCALE_CONVERSION)
            preprocessor.add_preprocessing_step(PreprocessingMethod.CONTRAST_ENHANCEMENT, alpha=1.4)
            preprocessor.add_preprocessing_step(PreprocessingMethod.NOISE_REDUCTION, method='gaussian')
            preprocessor.add_preprocessing_step(PreprocessingMethod.BINARIZATION, adaptive=True)
            
        elif document_type == 'receipt':
            preprocessor.add_preprocessing_step(PreprocessingMethod.GRAYSCALE_CONVERSION)
            preprocessor.add_preprocessing_step(PreprocessingMethod.SHADOW_REMOVAL)
            preprocessor.add_preprocessing_step(PreprocessingMethod.CONTRAST_ENHANCEMENT, alpha=1.5)
            preprocessor.add_preprocessing_step(PreprocessingMethod.NOISE_REDUCTION, method='median')
            preprocessor.add_preprocessing_step(PreprocessingMethod.BINARIZATION, adaptive=True, c=5)
            
        return preprocessor
