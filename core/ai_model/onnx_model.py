import cv2
from .base_model import BaseModel
from config.config import get_settings
from typing import Dict, List


class OnnxModel(BaseModel):
    """
    ONNX model class for object detection.
    """
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        super().__init__(model_path or get_settings().ID_DETECTOR_MODEL_PATH)
        self.model = None

    def load_model(self, num_class: int = None) -> None:
        """
        Load the ONNX model from the specified path.
        """
                    # Load model and set backend/target (this is customizable)
        self.model = cv2.dnn.readNetFromONNX(self.model_path)
        
    def predict(self, image_ndarray) -> List[Dict]:
        """
        Make a prediction on the given image.

        :param image: Path to the image file.
        :return: List of predictions.
        """
        self.model.setInput(image_ndarray)
        result = self.model.forward()
        return result