import numpy as np

class GENERAL_ROI:
    def __init__(self, image: np.ndarray = None):
        self.image = image
        
        if image is not None and not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

    def roi_extraction(self) -> dict:
        """
        Returns the original image without any ROI extraction In case there is no ROI extraction for this type.
        """
        if self.image is None:
            raise ValueError("No image provided")
            
        return {
            "image": self.image,
            "detected_parts": [{"general": [0, 0, self.image.shape[1], self.image.shape[0]]}]
        }
