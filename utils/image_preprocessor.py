from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
import math
from typing import Union, Tuple, Optional, List, Dict, Any, Callable
from utils.logging_utils import logger, exception_handler
from functools import lru_cache
import os
import threading

class ImagePreprocessor:
    """
    Advanced image processing utilities optimized for OCR performance.
    
    This class provides methods for enhancing document images to improve
    OCR accuracy through adaptive algorithms and multi-stage processing.
    """
    
    # Constants for OpenCV interpolation methods
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_AREA = cv2.INTER_AREA
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
    
    # Constants for thresholding
    THRESH_BINARY = cv2.THRESH_BINARY
    THRESH_BINARY_INV = cv2.THRESH_BINARY_INV
    THRESH_OTSU = cv2.THRESH_OTSU
    
    # Constants for color conversion
    COLOR_GRAY2RGB = cv2.COLOR_GRAY2RGB
    COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
    COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
    COLOR_RGB2LAB = cv2.COLOR_RGB2LAB
    COLOR_LAB2RGB = cv2.COLOR_LAB2RGB
    
    # Constants for morphological operations
    MORPH_RECT = cv2.MORPH_RECT
    MORPH_ELLIPSE = cv2.MORPH_ELLIPSE
    MORPH_CROSS = cv2.MORPH_CROSS
    
    # Thread local storage for caching
    _thread_local = threading.local()

    @staticmethod
    def get_optimal_kernel_size(image: np.ndarray) -> int:
        """
        Dynamically determine the optimal kernel size based on image dimensions.
        
        Args:
            image: Input image
            
        Returns:
            Optimal kernel size (odd number)
        """
        # Calculate based on image dimensions
        height, width = image.shape[:2]
        avg_dimension = (height + width) / 2
        
        # Scale kernel size with image dimensions, keeping it odd
        kernel_size = max(3, min(25, int(avg_dimension / 100) * 2 + 1))
        return kernel_size

    @staticmethod
    @exception_handler
    def noise_removal(image: np.ndarray, ksize: Optional[int] = None) -> np.ndarray:
        """
        Remove noise from an image using adaptive methods based on image content.
        
        Args:
            image: Input image
            ksize: Optional kernel size (auto-calculated if None)
            
        Returns:
            Denoised image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Auto-detect appropriate kernel size
        if ksize is None:
            ksize = ImagePreprocessor.get_optimal_kernel_size(image)
            
        if ksize % 2 == 0:
            ksize = ksize + 1  # Ensure kernel size is odd
        
        # Choose appropriate denoising method based on image characteristics
        gray = ImagePreprocessor.convert_to_grayscale(image.copy())
        noise_level = np.std(gray)
        
        if noise_level > 30:  # High noise
            # Non-local means denoising for high noise
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) if len(image.shape) > 2 else cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            # Gaussian blur for low noise
            return cv2.GaussianBlur(image, (ksize, ksize), 0)

    @staticmethod
    @exception_handler
    def adaptive_thresholding(
        image: np.ndarray, 
        block_size: Optional[int] = None, 
        c: int = 2,
        method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    ) -> np.ndarray:
        """
        Apply adaptive thresholding optimized for document images.
        
        Args:
            image: Input grayscale image
            block_size: Size of pixel neighborhood (auto-calculated if None)
            c: Constant subtracted from mean
            method: Thresholding method
            
        Returns:
            Binary image after adaptive thresholding
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, ImagePreprocessor.COLOR_RGB2GRAY)
        
        # Auto-detect block size based on image dimensions
        if block_size is None:
            height, width = image.shape[:2]
            block_size = max(3, min(51, int(min(height, width) / 30) * 2 + 1))
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size = block_size + 1
            
        return cv2.adaptiveThreshold(
            image, 
            255, 
            method,
            cv2.THRESH_BINARY, 
            block_size, 
            c
        )

    @staticmethod
    @exception_handler
    def binarization(
        image: np.ndarray, 
        threshold: int = 128, 
        max_value: int = 255, 
        threshold_type: int = cv2.THRESH_BINARY, 
        inverted: bool = False, 
        otsu: bool = False
    ) -> np.ndarray:
        """
        Apply advanced binarization with automatic parameter selection.
        
        Args:
            image: Input grayscale image
            threshold: Threshold value
            max_value: Maximum value after thresholding
            threshold_type: Type of thresholding
            inverted: Whether to invert the threshold
            otsu: Whether to use Otsu's method
            
        Returns:
            Binary image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, ImagePreprocessor.COLOR_RGB2GRAY)
            
        # Apply dynamic histogram equalization to improve contrast before binarization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
            
        # Apply threshold type and inversion if needed
        if inverted:
            threshold_type = threshold_type | cv2.THRESH_BINARY_INV
            
        if otsu:
            threshold_type = threshold_type | cv2.THRESH_OTSU
            threshold = 0
            
        # Apply thresholding
        _, binary_image = cv2.threshold(
            image, threshold, max_value, threshold_type
        )
        
        # Apply morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        
        return binary_image

    @staticmethod
    @exception_handler
    def deskew(image: np.ndarray, angle_range: Tuple[int, int] = (-15, 15)) -> np.ndarray:
        """
        Advanced deskewing to correct document orientation.
        
        Uses multiple methods to detect skew angles and applies the most reliable correction.
        
        Args:
            image: Input image
            angle_range: Range of angles to check for skew
            
        Returns:
            Deskewed image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Keep original for comparison
        original = image.copy()
            
        # Convert to grayscale if needed
        gray = image.copy()
        if len(gray.shape) > 2:
            gray = cv2.cvtColor(gray, ImagePreprocessor.COLOR_RGB2GRAY)
            
        # Apply thresholding
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # Method 1: Using contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to exclude noise
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if contours:
            # Find the rotated rectangles
            angles = []
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                # Adjust angle to be within range [-45, 45]
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                    
                # Only consider angles within our range
                if angle_range[0] <= angle <= angle_range[1]:
                    angles.append(angle)
            
            if angles:
                # Use the median angle to avoid outliers
                skew_angle = np.median(angles)
                
                # Only deskew if angle is significant
                if abs(skew_angle) > 0.5:
                    # Rotate the image to correct the skew
                    height, width = image.shape[:2]
                    center = (width/2, height/2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    
                    logger.debug(f"Image deskewed by {skew_angle:.2f} degrees")
                    return rotated
        
        # Method 2: Using Hough Lines if contour method failed
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:  # Avoid division by zero
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # Normalize angle
                    if angle < -45:
                        angle = 90 + angle
                    elif angle > 45:
                        angle = angle - 90
                    
                    # Only consider angles within range
                    if angle_range[0] <= angle <= angle_range[1]:
                        angles.append(angle)
            
            if angles:
                # Use the median angle to avoid outliers
                skew_angle = np.median(angles)
                
                # Only deskew if angle is significant
                if abs(skew_angle) > 0.5:
                    height, width = image.shape[:2]
                    center = (width/2, height/2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    
                    logger.debug(f"Image deskewed by {skew_angle:.2f} degrees using Hough method")
                    return rotated
        
        # Return original if deskewing wasn't needed or failed
        return original

    @staticmethod
    @exception_handler
    def enhance_contrast(
        image: np.ndarray, 
        clip_limit: float = 2.0, 
        tile_grid_size: Tuple[int, int] = (8, 8),
        method: str = "clahe"
    ) -> np.ndarray:
        """
        Enhance contrast in an image using multiple available methods.
        
        Args:
            image: Input image
            clip_limit: Contrast limit for CLAHE
            tile_grid_size: Size of the grid for CLAHE
            method: Enhancement method ('clahe', 'histogram', 'adaptive')
            
        Returns:
            Contrast-enhanced image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        
        # Select appropriate enhancement method
        if method == "clahe":
            # Process based on color or grayscale
            if len(image.shape) > 2:  # Color image
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                
                # Split channels
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                cl = clahe.apply(l)
                
                # Merge channels back
                enhanced_lab = cv2.merge((cl, a, b))
                
                # Convert back to original color space
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:  # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced = clahe.apply(image)
                
        elif method == "histogram":
            # Simple histogram equalization
            if len(image.shape) > 2:  # Color image
                # Apply to each channel separately
                channels = cv2.split(image)
                eq_channels = [cv2.equalizeHist(ch) for ch in channels]
                enhanced = cv2.merge(eq_channels)
            else:  # Grayscale image
                enhanced = cv2.equalizeHist(image)
                
        elif method == "adaptive":
            # Adaptive histogram equalization with custom parameters
            if len(image.shape) > 2:  # Color image
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply adaptive enhancement to L channel
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                cl = clahe.apply(l)
                
                # Apply additional processing for documents
                kernel = np.ones((3, 3), np.uint8)
                cl = cv2.morphologyEx(cl, cv2.MORPH_CLOSE, kernel)
                
                # Merge channels back
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                # For grayscale
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced = clahe.apply(image)
                kernel = np.ones((3, 3), np.uint8)
                enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        else:
            # Default to original if method not recognized
            enhanced = image.copy()
        
        return enhanced

    @staticmethod
    @exception_handler
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale with optimized handling of different color spaces.
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Check if image is already grayscale
        if len(image.shape) <= 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            return image.copy()
        
        # Detect color space and use appropriate conversion
        if image.shape[2] == 3:  # RGB/BGR
            return cv2.cvtColor(image, ImagePreprocessor.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:  # RGBA/BGRA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            # Fallback for unusual color spaces
            return np.mean(image, axis=2).astype(np.uint8)

    @staticmethod
    @exception_handler
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert an image to RGB color space, handling multiple input formats.
        
        Args:
            image: Input image
            
        Returns:
            RGB image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Already RGB with 3 channels
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image.copy()
            
        # Grayscale to RGB
        if len(image.shape) <= 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            return cv2.cvtColor(image, ImagePreprocessor.COLOR_GRAY2RGB)
            
        # RGBA to RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Fallback for unusual formats
        return image

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_optimal_preprocessing_pipeline(image_shape: Tuple[int, int], doc_type: str = None) -> List[Dict[str, Any]]:
        """
        Get an optimized preprocessing pipeline based on image characteristics and document type.
        
        Args:
            image_shape: Shape of the image (height, width)
            doc_type: Optional document type for specialized processing
            
        Returns:
            List of preprocessing steps to apply
        """
        height, width = image_shape
        
        # Default pipeline
        pipeline = [
            {"method": "deskew", "params": {}},
            {"method": "enhance_contrast", "params": {"method": "clahe"}},
        ]
        
        # Add document-specific steps
        if doc_type == "passport" or doc_type == "mrz":
            # MRZ-specific processing
            pipeline.append({"method": "binarization", "params": {
                "threshold": 0, 
                "otsu": True, 
                "inverted": False
            }})
        elif doc_type == "id_card":
            # ID card specific processing
            pipeline.append({"method": "noise_removal", "params": {}})
            pipeline.append({"method": "binarization", "params": {
                "threshold": 110, 
                "inverted": True
            }})
        elif doc_type == "handwritten":
            # Handwritten text processing
            pipeline.append({"method": "noise_removal", "params": {"ksize": 5}})
            pipeline.append({"method": "adaptive_thresholding", "params": {}})
        
        # Small image processing
        if height < 1000 or width < 1000:
            # Upscaling step
            pipeline.insert(0, {"method": "resize", "params": {
                "target_size": (int(width*1.5), int(height*1.5)),
                "interpolation": cv2.INTER_CUBIC
            }})
            
        return pipeline

    @staticmethod
    @exception_handler
    def auto_preprocess(image: np.ndarray, doc_type: str = None) -> np.ndarray:
        """
        Automatically preprocess an image using an optimized pipeline.
        
        Args:
            image: Input image
            doc_type: Optional document type for specialized processing
            
        Returns:
            Preprocessed image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Get shape for pipeline selection
        height, width = image.shape[:2]
        
        # Get optimized pipeline
        pipeline = ImagePreprocessor._get_optimal_preprocessing_pipeline((height, width), doc_type)
        
        # Apply pipeline steps
        processed = image.copy()
        for step in pipeline:
            method_name = step["method"]
            params = step["params"]
            
            try:
                # Get the preprocessing method
                method = getattr(ImagePreprocessor, method_name)
                # Apply the method with parameters
                processed = method(processed, **params)
            except Exception as e:
                logger.warning(f"Failed to apply {method_name}: {str(e)}")
        
        return processed

    @staticmethod
    @exception_handler
    def resize(
        image: np.ndarray, 
        target_size: Union[Tuple[int, int], int] = None,
        scale: Optional[float] = None,
        interpolation: int = cv2.INTER_LINEAR,
        keep_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Resize an image to target dimensions with advanced options.
        
        Args:
            image: Input image
            target_size: Target size as (width, height) or single dimension
            scale: Scale factor for resizing
            interpolation: Interpolation method
            keep_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # If scale is provided, perform resizing based on the scale factor
        if scale is not None:
            h, w = image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Handle different input formats
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
            
        # Original dimensions
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # If keeping aspect ratio
        if keep_aspect_ratio:
            # Calculate scale to maintain aspect ratio
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Choose best interpolation method based on scaling direction
            if scale > 1:  # Upscaling
                interp = cv2.INTER_CUBIC
            else:  # Downscaling
                interp = cv2.INTER_AREA
            
            # Resize with computed dimensions
            resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
            
            # Create a blank canvas of target size
            if len(image.shape) > 2:
                result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            else:
                result = np.zeros((target_h, target_w), dtype=image.dtype)
                
            # Paste the resized image at the center
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            if len(image.shape) > 2:
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
            else:
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
            return result
        else:
            # Choose best interpolation method if not specified
            if interpolation == cv2.INTER_LINEAR:
                if target_w > w and target_h > h:  # Upscaling
                    interpolation = cv2.INTER_CUBIC
                elif target_w < w and target_h < h:  # Downscaling
                    interpolation = cv2.INTER_AREA
                    
            # Simple resize without maintaining aspect ratio
            return cv2.resize(image, target_size, interpolation=interpolation)

    @staticmethod
    @exception_handler
    def expand_image(
        image_dim: Union[List[float], np.ndarray, Tuple[float, float, float, float]], 
        scale_height: float = 1.2, 
        scale_width: float = 1.0, 
        image_shape: Optional[Tuple[int, int]] = None
    ) -> List[int]:
        """
        Calculate expanded bounding box dimensions with intelligent constraints.
        
        Args:
            image_dim: Original bounding box dimensions [x1, y1, x2, y2]
            scale_height: Factor to scale height by
            scale_width: Factor to scale width by
            image_shape: Shape of the image to constrain coordinates (height, width)
            
        Returns:
            Expanded bounding box as [x1, y1, x2, y2]
        """
        # Extract coordinates
        x1, y1, x2, y2 = image_dim
        
        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate original dimensions
        width = x2 - x1
        height = y2 - y1

        # Calculate new dimensions with minimum size constraints
        new_height = max(int(height * scale_height), 10)  # Minimum height of 10px
        new_width = max(int(width * scale_width), 10)     # Minimum width of 10px

        # Calculate new bounding box coordinates
        new_x1 = max(int(center_x - new_width / 2), 0)
        new_y1 = max(int(center_y - new_height / 2), 0)
        
        # Handle image boundaries if shape is provided
        if image_shape:
            img_height, img_width = image_shape
            new_x2 = min(int(center_x + new_width / 2), img_width - 1)
            new_y2 = min(int(center_y + new_height / 2), img_height - 1)
        else:
            new_x2 = int(center_x + new_width / 2)
            new_y2 = int(center_y + new_height / 2)
        
        # Ensure box has positive dimensions
        if new_x1 >= new_x2:
            new_x2 = new_x1 + 1
        if new_y1 >= new_y2:
            new_y2 = new_y1 + 1

        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    @exception_handler
    def expand_image_background(
        image: np.ndarray, 
        scale_height: float = 5.0, 
        scale_width: float = 2.0, 
        background_color: Union[Tuple[int, int, int], int] = None,
        position: str = "center"
    ) -> np.ndarray:
        """
        Expand an image by adding padding/background with positioning options.
        
        Args:
            image: Input image
            scale_height: Factor to scale height by
            scale_width: Factor to scale width by
            background_color: Color of the background
            position: Positioning of the original image ('center', 'top', 'bottom', 'left', 'right')
            
        Returns:
            Expanded image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Calculate new dimensions
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_height), int(w * scale_width)
        
        # Create new image with background color
        channels = 3 if len(image.shape) > 2 else 1
        dtype = image.dtype
        
        if background_color is None:
            background_color = tuple(image[0, 0]) if channels == 3 else image[0, 0]
        
        if channels == 1:
            bg_color = background_color[0] if isinstance(background_color, tuple) else background_color
            expanded = np.full((new_h, new_w), bg_color, dtype=dtype)
        else:
            expanded = np.full((new_h, new_w, channels), background_color, dtype=dtype)
            
        # Calculate position to place original image
        if position == "center":
            y_offset = (new_h - h) // 2
            x_offset = (new_w - w) // 2
        elif position == "top":
            y_offset = 0
            x_offset = (new_w - w) // 2
        elif position == "bottom":
            y_offset = new_h - h
            x_offset = (new_w - w) // 2
        elif position == "left":
            y_offset = (new_h - h) // 2
            x_offset = 0
        elif position == "right":
            y_offset = (new_h - h) // 2
            x_offset = new_w - w
        else:
            # Default to center
            y_offset = (new_h - h) // 2
            x_offset = (new_w - w) // 2
        
        # Place original image at the calculated position
        if channels == 1:
            expanded[y_offset:y_offset+h, x_offset:x_offset+w] = image
        else:
            expanded[y_offset:y_offset+h, x_offset:x_offset+w, :] = image
            
        return expanded

    @staticmethod
    @exception_handler  
    def cast_and_scale(
        image: np.ndarray, 
        cast_type: str = "float32", 
        scale_factor: float = 1.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Cast an image to a different data type and scale its values.
        
        Args:
            image: Input image
            cast_type: Target data type ("float32", "uint8", etc.)
            scale_factor: Factor to scale pixel values by
            normalize: Whether to normalize float images to [0,1]
            
        Returns:
            Cast and scaled image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Convert data type with appropriate scaling
        if cast_type == "float32":
            casted = image.astype(np.float32)
            
            # Scale values to [0, 1] for float32
            if normalize and casted.max() > 1.0:
                casted = casted / 255.0
                
            # Apply additional scaling if needed
            if scale_factor != 1.0:
                casted = casted * scale_factor
                
        elif cast_type == "uint8":
            # Handle float images with values in [0, 1]
            if issubclass(image.dtype.type, np.floating) and image.max() <= 1.0:
                scaled = image * 255
            else:
                scaled = image * scale_factor
                
            # Clip values to valid range
            casted = np.clip(scaled, 0, 255).astype(np.uint8)
            
        else:
            # For other types, cast and scale
            casted = (image * scale_factor).astype(getattr(np, cast_type))
            
        return casted

    @staticmethod
    @exception_handler
    def reshape(
        image: np.ndarray, 
        image_shape: Tuple[int, ...],
        allow_copy: bool = True
    ) -> np.ndarray:
        """
        Reshape an image to target dimensions with safety checks.
        
        Args:
            image: Input image
            image_shape: Target shape
            allow_copy: Whether to allow copying to ensure contiguity
            
        Returns:
            Reshaped image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        try:
            # Ensure array is contiguous for safe reshaping
            if not image.flags.contiguous and allow_copy:
                image = np.ascontiguousarray(image)
                
            # Validate total size remains the same
            current_size = np.prod(image.shape)
            target_size = np.prod(image_shape)
            
            if current_size != target_size:
                raise ValueError(f"Cannot reshape array of size {current_size} to shape {image_shape} with size {target_size}")
                
            return image.reshape(image_shape)
        except Exception as e:
            logger.error(f"Reshape failed: {str(e)}")
            raise ValueError(f"Cannot reshape image from {image.shape} to {image_shape}: {str(e)}")

    @staticmethod
    @exception_handler
    def remove_borders(
        image: np.ndarray, 
        threshold: int = 200, 
        padding: int = 5,
        min_content_area: float = 0.01  # Minimum percentage of content relative to total area
    ) -> np.ndarray:
        """
        Remove borders/frames from an image with intelligent content detection.
        
        Args:
            image: Input image
            threshold: Threshold for detecting borders
            padding: Padding to add around detected content
            min_content_area: Minimum content area ratio to proceed with cropping
            
        Returns:
            Image with borders removed
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        total_area = orig_h * orig_w
            
        # Convert to grayscale if needed
        gray = image.copy()
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Apply binary threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find non-zero pixels (content)
        non_zero = cv2.findNonZero(binary)
        
        if non_zero is None or len(non_zero) == 0:
            logger.warning("No content detected in the image")
            return image
            
        # Get the bounding box of content
        x, y, w, h = cv2.boundingRect(non_zero)
        
        # Verify the content area is reasonable (not too small)
        content_area = w * h
        if content_area < min_content_area * total_area:
            logger.info(f"Detected content area too small ({content_area/total_area:.2%}), keeping original image")
            return image
            
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(orig_w - x, w + 2 * padding)
        h = min(orig_h - y, h + 2 * padding)
        
        # Crop the image to the content
        return image[y:y+h, x:x+w]

    @staticmethod
    def document_enhancement_pipeline(
        image: np.ndarray, 
        doc_type: str = None
    ) -> np.ndarray:
        """
        Complete document enhancement pipeline optimized for OCR.
        
        Args:
            image: Input document image
            doc_type: Document type for specialized processing
            
        Returns:
            Enhanced document image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")

        try:
            # Start with a copy of the original image
            processed = image.copy()
            
            # 1. Remove borders/extra margin
            processed = ImagePreprocessor.remove_borders(processed, padding=10)
            
            # 2. Deskew to correct orientation
            processed = ImagePreprocessor.deskew(processed)
            
            # 3. Enhance contrast
            processed = ImagePreprocessor.enhance_contrast(processed, method="adaptive")
            
            # 4. Remove noise
            processed = ImagePreprocessor.noise_removal(processed)
            
            # 5. Document type specific processing
            if doc_type:
                if doc_type.lower() in ["passport", "mrz"]:
                    # Optimize for MRZ reading
                    processed = ImagePreprocessor.binarization(processed, threshold=0, otsu=True)
                elif doc_type.lower() in ["id_card", "license", "card"]:
                    # ID card specific processing
                    processed = ImagePreprocessor.binarization(processed, threshold=110, inverted=True)
                elif doc_type.lower() in ["receipt", "invoice"]:
                    # Receipt specific processing
                    processed = ImagePreprocessor.adaptive_thresholding(processed)
            
            logger.info(f"Document enhancement pipeline completed for type: {doc_type}")
            return processed
            
        except Exception as e:
            logger.error(f"Document enhancement failed: {str(e)}")
            # Return original image if enhancement fails
            return image

    @staticmethod
    @exception_handler
    def batch_process(
        images: List[np.ndarray], 
        pipeline: List[Dict[str, Any]],
        max_workers: int = 4
    ) -> List[np.ndarray]:
        """
        Process multiple images in parallel using the specified pipeline.
        
        Args:
            images: List of input images
            pipeline: List of processing steps to apply
            max_workers: Maximum number of worker threads
            
        Returns:
            List of processed images
        """
        if not images:
            return []
            
        # Define a worker function for parallel processing
        def process_single_image(image):
            processed = image.copy()
            for step in pipeline:
                method_name = step.get('method')
                params = step.get('params', {})
                
                # Get the method from ImagePreprocessor
                method = getattr(ImagePreprocessor, method_name, None)
                if method:
                    try:
                        processed = method(processed, **params)
                    except Exception as e:
                        logger.warning(f"Failed to apply {method_name}: {str(e)}")
                        
            return processed
        
        # Process images in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_image, images))
            
        return results

    @staticmethod
    @exception_handler
    def document_enhancement_for_batch(
        images: List[np.ndarray], 
        doc_type: str = None
    ) -> List[np.ndarray]:
        """
        Enhanced batch document processing with parallel execution.
        
        Args:
            images: List of document images to enhance
            doc_type: Type of document for specialized processing
            
        Returns:
            List of enhanced document images
        """
        if not images or len(images) == 0:
            return []
            
        # Get optimal pipeline for the document type
        if len(images[0].shape) >= 2:
            height, width = images[0].shape[:2]
            pipeline = ImagePreprocessor._get_optimal_preprocessing_pipeline((height, width), doc_type)
        else:
            pipeline = []
            
        # Process images in parallel
        return ImagePreprocessor.batch_process(images, pipeline)