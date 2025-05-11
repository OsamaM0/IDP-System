import numpy as np
import cv2
from utils.image_preprocessor import ImagePreprocessor
from core.preprocessor.base_preprocessor import BasePreprocessor
from utils.logging_utils import logger, exception_handler

class MRZPreprocessor(BasePreprocessor):
    """
    Specialized preprocessor for Machine Readable Zones (MRZ) in passports and travel documents.
    Enhances MRZ text for better OCR recognition by applying specialized filters and transformations.
    """
    
    @exception_handler
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Process the MRZ region of an image to optimize OCR accuracy.
        
        Args:
            image: Input image containing MRZ region
            
        Returns:
            Processed image optimized for MRZ OCR
        """
        logger.info("Applying specialized MRZ preprocessing")
        
        # Ensure input is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        
        # Convert to grayscale if not already
        if len(image.shape) > 2:
            gray = ImagePreprocessor.convert_to_grayscale(image)
        else:
            gray = image.copy()
        
        # Apply noise reduction with a bilateral filter (preserves edges better than gaussian)
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE to improve local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding optimized for MRZ font
        binary = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            13,  # Block size tuned for MRZ font
            7     # Constant tuned for MRZ contrast
        )
        
        # Remove small noise artifacts while preserving text
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Convert back to RGB
        processed_image = ImagePreprocessor.convert_to_rgb(morph)
        
        logger.info("MRZ preprocessing completed successfully")
        return processed_image
