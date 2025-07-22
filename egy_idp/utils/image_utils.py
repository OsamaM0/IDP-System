"""
Image utility functions for the IDP System.

This module contains utility functions for image processing, conversion, and manipulation
used throughout the IDP system.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Union, Tuple, Optional


def read_image(
    image_path: str = None, 
    image_bytes: bytes = None, 
    image_array: np.ndarray = None
) -> Image.Image:
    """
    Read an image from various sources and return as PIL Image.
    
    Args:
        image_path: Path to the image file
        image_bytes: Image data as bytes
        image_array: Image as numpy array
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If no valid input is provided
    """
    if image_bytes is not None:
        return Image.open(io.BytesIO(image_bytes))
    elif image_path is not None:
        return Image.open(image_path)
    elif image_array is not None:
        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        if len(image_array.shape) == 3:
            # Color image (BGR to RGB)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_array)
    else:
        raise ValueError("No valid input provided. Use image_path, image_bytes, or image_array.")


def numpy_to_bytes(image_array: np.ndarray, format: str = 'PNG') -> bytes:
    """
    Convert numpy array to image bytes.
    
    Args:
        image_array: Image as numpy array
        format: Output format ('PNG', 'JPEG', etc.)
        
    Returns:
        Image data as bytes
    """
    # Ensure proper data type
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Convert BGR to RGB if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image and save to bytes
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return buffer.getvalue()


def image_to_base64(image: Union[np.ndarray, Image.Image], format: str = 'PNG') -> str:
    """
    Convert image to base64 string.
    
    Args:
        image: Image as numpy array or PIL Image
        format: Output format
        
    Returns:
        Base64 encoded string
    """
    if isinstance(image, np.ndarray):
        image_bytes = numpy_to_bytes(image, format)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        image_bytes = buffer.getvalue()
    
    return base64.b64encode(image_bytes).decode('utf-8')


def draw_bounding_box(
    image: np.ndarray, 
    bbox: Tuple[int, int, int, int], 
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = None
) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: Image as numpy array
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        color: Box color (BGR)
        thickness: Line thickness
        label: Optional label to draw
        
    Returns:
        Image with bounding box drawn
    """
    image_copy = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            image_copy, 
            (x1, y1 - label_size[1] - 10), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        cv2.putText(
            image_copy, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
    
    return image_copy


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Basic image preprocessing for OCR.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    return processed


def rotate_card(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by the specified angle.
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    
    # Adjust translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated


# Compatibility classes for backwards compatibility
class ImagePreprocessor:
    """Image preprocessing utility class."""
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """Enhance image contrast."""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values."""
        if image.dtype != np.uint8:
            image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            return image_norm.astype(np.uint8)
        return image
    
    @staticmethod
    def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to specified dimensions."""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def apply_threshold(image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
        """Apply binary threshold to image."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        return thresholded
    
    @staticmethod
    def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
        """Apply adaptive threshold to image."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )


def resize_image(
    image: np.ndarray, 
    width: int = None, 
    height: int = None, 
    maintain_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image to specified dimensions.
    
    Args:
        image: Input image
        width: Target width
        height: Target height
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if maintain_aspect_ratio:
        if width is not None and height is not None:
            # Calculate which dimension to use based on aspect ratio
            aspect_ratio = w / h
            if (width / height) > aspect_ratio:
                width = int(height * aspect_ratio)
            else:
                height = int(width / aspect_ratio)
        elif width is not None:
            aspect_ratio = w / h
            height = int(width / aspect_ratio)
        elif height is not None:
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image: Input image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        Enhanced image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def normalize_image(image: np.ndarray) -> np.ndarray:
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


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image to specified bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def get_image_dimensions(image: Union[np.ndarray, Image.Image]) -> Tuple[int, int]:
    """
    Get image dimensions (width, height).
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (width, height)
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        return w, h
    else:
        return image.size


def convert_color_space(image: np.ndarray, conversion: int) -> np.ndarray:
    """
    Convert image between color spaces.
    
    Args:
        image: Input image
        conversion: OpenCV color conversion code (e.g., cv2.COLOR_BGR2RGB)
        
    Returns:
        Converted image
    """
    return cv2.cvtColor(image, conversion)
