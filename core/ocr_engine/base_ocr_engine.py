from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from utils.logging_utils import log_execution_time, exception_handler
from core.exceptions import OCREngineException
import queue

@dataclass
class BoundingBox:
    """
    Represents a bounding box for OCR results.
    
    Attributes:
        x1: Top-left x-coordinate
        y1: Top-left y-coordinate
        x2: Bottom-right x-coordinate
        y2: Bottom-right y-coordinate
    """
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }
    
    def expand(self, scale_x: float = 1.0, scale_y: float = 1.0) -> "BoundingBox":
        """
        Expand the bounding box by the given scale factors.
        
        Args:
            scale_x: Horizontal scale factor
            scale_y: Vertical scale factor
            
        Returns:
            A new BoundingBox with expanded dimensions
        """
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("Scale factors must be positive values")
            
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2
        
        new_width = width * scale_x
        new_height = height * scale_y
        
        new_x1 = max(0, int(center_x - new_width / 2))
        new_y1 = max(0, int(center_y - new_height / 2))
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)
        
        return BoundingBox(
            x1=new_x1,
            y1=new_y1,
            x2=new_x2,
            y2=new_y2
        )
        
    def is_valid(self) -> bool:
        """
        Check if the bounding box is valid (positive dimensions).
        
        Returns:
            bool: True if valid, False otherwise
        """
        return (self.x1 < self.x2 and 
                self.y1 < self.y2 and 
                self.x1 >= 0 and 
                self.y1 >= 0)
    
    @property
    def width(self) -> int:
        """Get the width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get the height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Get the area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the bounding box."""
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))


@dataclass
class OCRResult:
    """
    Represents an OCR result with text and bounding box.
    
    Attributes:
        text: The extracted text
        bbox: The bounding box containing the text
        confidence: Confidence score (0.0 to 1.0)
        language: Detected language
    """
    text: str
    bbox: BoundingBox
    confidence: float = 1.0
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "language": self.language
        }


class OCREngineInterface(ABC):
    """Base interface for all OCR engines."""
    
    @abstractmethod
    def __init__(self, languages: List, **kwargs):
        """
        Initialize the OCR engine with the specified languages.
        
        Args:
            languages: List of languages to support
            **kwargs: Additional keyword arguments for the OCR engine
        """
        pass
    
    @abstractmethod
    @log_execution_time
    @exception_handler
    def get_text(self, image: np.ndarray) -> str:
        """
        Extract text from an image.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Extracted text as string
            
        Raises:
            OCREngineException: If text extraction fails
        """
        pass
    
    @abstractmethod
    @log_execution_time
    @exception_handler
    def get_text_with_bounding_boxes(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text with bounding box information.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            List of OCRResult objects containing text and bounding boxes
            
        Raises:
            OCREngineException: If text extraction fails
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image before OCR.
        
        Args:
            image: Input image as NumPy array
            
        Returns:
            Preprocessed image
        """
        # Default implementation returns the image unchanged
        # Subclasses can override this method to implement custom preprocessing
        return image
    
    def postprocess_text(self, text: str) -> str:
        """
        Postprocess the extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            Postprocessed text
        """
        # Default implementation returns the text unchanged
        # Subclasses can override this method to implement custom postprocessing
        return text
    
    @staticmethod
    def _convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale if it's not already.
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) > 2 and image.shape[2] > 1:
            import cv2
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def _enhance_image(image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better OCR results.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        import cv2
        
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Apply noise removal
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable objects (e.g., instances of queue.SimpleQueue)
        for key, value in list(state.items()):
            if isinstance(value, queue.SimpleQueue):
                del state[key]
        return state