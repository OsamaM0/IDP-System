from ultralytics import YOLO
from core.ai_model.base_model import BaseModel
from config.config import get_settings
from core.ai_model.model_type_enums import ModelType
from typing import Dict, List


class YoloModel(BaseModel):
    """
    YOLO model class for object detection.
    """
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        super().__init__(model_path or get_settings().ID_DETECTOR_MODEL_PATH)
        self.model = None

    def load_model(self, class_num: int = None) -> None:
        """
        Load the YOLO model from the specified path.
        """
        self.model = YOLO(self.model_path)

    def predict(self, image_ndarray) -> List[Dict]:
        """
        Make a prediction on the given image.

        :param image: Path to the image file.
        :return: List of predictions.
        """
        results = self.model(image_ndarray, device=self.device)
        return results