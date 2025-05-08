from utils.image_preprocessor import ImagePreprocessor
from .base_preprocessor import BasePreprocessor
import numpy as np

class TaxPreprocessor(BasePreprocessor):
    def preprocess(self, image):
        # Ensure input is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        
        # Ensure the image is grayscale before binarization
        if len(image.shape) > 2:
            image = ImagePreprocessor.convert_to_grayscale(image)
        
        # Apply binarization methods
        processed_image = ImagePreprocessor.binarization(image, threshold=155, inverted=True)
        processed_image = ImagePreprocessor.binarization(processed_image, threshold=0, max_value=255, otsu=True, inverted=True)
        
        # Convert to RGB
        processed_image = ImagePreprocessor.convert_to_rgb(processed_image)
        
        return processed_image
