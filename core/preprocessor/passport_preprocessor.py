import numpy as np
from utils.image_preprocessor import ImagePreprocessor
from .base_preprocessor import BasePreprocessor

class PassportPreprocessor(BasePreprocessor):
    
    def preprocess(self, image):
        # Ensure input is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        # Apply resizing to the image
        image = ImagePreprocessor.resize(cropped_image=image, image_shape=(256, 256), interpolation=ImagePreprocessor.INTER_NEAREST)
        
        # Apply casting to float32
        image = ImagePreprocessor.cast_and_scale(cast_type="float32", image=image)
        
        # Apply reshaping to the image
        if len(image.shape) >= 3:
            image = image[:, :, :3]
        image = ImagePreprocessor.reshape(image=image, image_shape=(1, 256, 256, 3))
        
        return image
