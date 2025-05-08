from abc import ABC, abstractmethod
import numpy as np

class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Process the input image and return the result"""
        pass
