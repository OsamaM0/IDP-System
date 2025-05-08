from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

@dataclass
class OCRResult:
    text: str
    bbox: BoundingBox
    metadata: Optional[Dict] = None

class OCREngineInterface(ABC):
    """Base interface for all OCR engines"""
        
    @abstractmethod
    def get_text(self, image: np.ndarray) -> List[str]:
        """Extract text from image"""
        pass
    
    @abstractmethod
    def get_bboxes(self, image: np.ndarray) -> List[BoundingBox]:
        """Get bounding boxes for text regions"""
        pass
    
    @abstractmethod
    def get_text_with_bbox(self, image: np.ndarray) -> List[OCRResult]:
        """Get text with corresponding bounding boxes"""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass