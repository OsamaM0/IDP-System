from abc import ABC, abstractmethod
import numpy as np
from ..ai_model.model_type_enums import ModelType
class BaseROI(ABC):
    """
    Abstract base class for Region of Interest (ROI) Extraction.
    """
    def __init__(self, image: np.ndarray = None):
        self.image = image
        self.model = None
        if image is not None and not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
    
    @abstractmethod
    def roi_extraction(self) -> list[dict[str, list[int]]]:
        """Extract the region of interest from the image."""
        pass
