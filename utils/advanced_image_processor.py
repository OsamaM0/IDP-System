import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import threading
from functools import lru_cache
from utils.logging_utils import logger, exception_handler

class AdvancedImageProcessor:
    """
    Advanced image processing utilities specifically optimized for document OCR.
    
    This class provides methods to enhance document images before OCR processing,
    significantly improving text recognition accuracy.
    """
    
    # Thread-local storage for caching within individual threads
    _thread_local = threading.local()
    
    # Constants for morphology operations
    MORPH_RECT = cv2.MORPH_RECT
    MORPH_ELLIPSE = cv2.MORPH_ELLIPSE
    MORPH_CROSS = cv2.MORPH_CROSS
    
    @staticmethod
    @exception_handler
    def remove_background(image: np.ndarray) -> np.ndarray:
        """
        Remove complex backgrounds to isolate text.
        
        Args:
            image: Input image
            
        Returns:
            Image with background removed
        """
        if len(image.shape) == 2 or image.shape[2] == 1:
            # Already grayscale
            gray = image.copy() if len(image.shape) == 2 else image[:, :, 0]
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find large contours (potential background elements)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out small contours (likely text)
        mask = np.zeros_like(binary)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust threshold based on your specific needs
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
        # Invert mask to get text regions
        mask = 255 - mask
        
        # Apply mask to original image
        if len(image.shape) == 2:
            result = cv2.bitwise_and(image, image, mask=mask)
        else:
            result = cv2.bitwise_and(image, image, mask=mask)
            
        return result

    @staticmethod
    @exception_handler
    def shadow_removal(image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from document images.
        
        Args:
            image: Input image
            
        Returns:
            Image with shadows removed
        """
        if len(image.shape) == 2:
            # For grayscale images
            dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(image, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            return norm
        else:
            # For color images
            rgb_planes = cv2.split(image)
            result_planes = []
            
            for plane in rgb_planes:
                dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg = cv2.medianBlur(dilated, 21)
                diff = 255 - cv2.absdiff(plane, bg)
                norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(norm)
                
            result = cv2.merge(result_planes)
            return result

    @staticmethod
    @exception_handler
    def deskew_optimized(image: np.ndarray) -> np.ndarray:
        """
        Deskew document images with advanced algorithm.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get a binary image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            # Fallback to alternative method if no lines detected
            return AdvancedImageProcessor._deskew_alternative(image)
            
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            if x2 - x1 != 0:  # Avoid division by zero
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                # Normalize angle to -45 to 45 range
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                angles.append(angle)
        
        if not angles:
            return image
            
        # Use median angle to avoid outliers
        median_angle = np.median(angles)
        
        # Only deskew if angle is significant
        if abs(median_angle) > 0.5:
            h, w = image.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
        
    @staticmethod
    def _deskew_alternative(image: np.ndarray) -> np.ndarray:
        """Alternative deskewing method when line detection fails."""
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Calculate skew using image moments
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Only deskew if angle is significant
        if abs(angle) > 0.5:
            h, w = image.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
            
        return image

    @staticmethod
    @exception_handler
    def denoise_document(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Apply document-specific denoising.
        
        Args:
            image: Input image
            strength: Denoising strength (higher = smoother but may blur text)
            
        Returns:
            Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 2:
            # For grayscale image
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        else:
            # For color image
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
            
    @staticmethod
    @exception_handler
    def sharpen_text(image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        Sharpen text in document images.
        
        Args:
            image: Input image
            alpha: Sharpening strength multiplier
            
        Returns:
            Sharpened image
        """
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * alpha
        
        # Apply kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Clip values to valid range for uint8
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    @staticmethod
    @lru_cache(maxsize=16)  # Cache results for performance
    def get_document_pipeline(doc_type: str) -> List[Dict[str, Any]]:
        """
        Get an optimized image processing pipeline based on document type.
        
        Args:
            doc_type: Document type (e.g., 'passport', 'id_card', 'invoice')
            
        Returns:
            List of processing steps to apply
        """
        # Default pipeline
        pipeline = [
            {"method": "denoise_document", "args": {"strength": 5}},
            {"method": "deskew_optimized", "args": {}}
        ]
        
        # Document type specific enhancements
        if doc_type.lower() in ['passport', 'mrz']:
            pipeline.extend([
                {"method": "sharpen_text", "args": {"alpha": 1.2}},
                {"method": "adaptive_threshold", "args": {"block_size": 11, "c": 2}}
            ])
        elif doc_type.lower() in ['id_card', 'nidf', 'nidb']:
            pipeline.extend([
                {"method": "shadow_removal", "args": {}},
                {"method": "sharpen_text", "args": {"alpha": 1.5}}
            ])
        elif doc_type.lower() in ['invoice', 'receipt']:
            pipeline.extend([
                {"method": "remove_background", "args": {}},
                {"method": "adaptive_threshold", "args": {"block_size": 15, "c": 3}}
            ])
        
        return pipeline
    
    @staticmethod
    @exception_handler
    def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
        """
        Apply adaptive thresholding optimized for document text.
        
        Args:
            image: Input image
            block_size: Size of pixel neighborhood for thresholding
            c: Constant subtracted from mean
            
        Returns:
            Binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
        
        return binary
        
    @staticmethod
    @exception_handler
    def process_document(image: np.ndarray, doc_type: str = 'general') -> np.ndarray:
        """
        Process document image with optimized pipeline based on document type.
        
        Args:
            image: Input document image
            doc_type: Document type for specialized processing
            
        Returns:
            Processed document image
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image provided")
            
        # Get the appropriate processing pipeline
        pipeline = AdvancedImageProcessor.get_document_pipeline(doc_type)
        
        # Process the image through the pipeline
        result = image.copy()
        for step in pipeline:
            method_name = step["method"]
            args = step["args"]
            
            # Get the processing method
            method = getattr(AdvancedImageProcessor, method_name)
            
            # Apply the method
            result = method(result, **args)
            
        return result
            
    @staticmethod
    @exception_handler
    def process_roi(image: np.ndarray, roi_type: str) -> np.ndarray:
        """
        Process a specific region of interest with optimized parameters.
        
        Args:
            image: Input ROI image
            roi_type: Type of ROI (e.g., 'text', 'mrz', 'number', 'photo')
            
        Returns:
            Processed ROI image
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image provided")
            
        # Process based on ROI type
        if roi_type == 'mrz':
            # MRZ-specific processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            # Adaptive threshold optimized for MRZ
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 13, 5)
            # MRZ often has horizontal text lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
            dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel, iterations=1)
            return dilated
            
        elif roi_type == 'number':
            # Number-specific processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            # Binary threshold with Otsu's method
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
            
        elif roi_type == 'photo':
            # Enhance photo region without binarization
            if len(image.shape) == 2:
                # Convert grayscale to color
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            # Apply denoising
            denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            return denoised
            
        else:  # Default text processing
            # General text processing
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            return binary
