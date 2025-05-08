from core.ai_model.deep_model import DeepModel
from core.ai_model.yolo_model import YoloModel
from core.ai_model.onnx_model import OnnxModel
from core.ai_model.model_type_enums import ModelType

class ModelFactory:
    """
    Factory class to create model objects based on the model type.
    """

    @staticmethod
    def create_model(model_name: ModelType, device: str = "cpu"):
        """
        Creates a model object based on the type of model.

        Args:
            model_name (ModelType): The model type (e.g., "DeepModel", "YoloModel", "OnnxModel").

        Raises:
            ValueError: If the model type is not supported.

        Returns:
            object: A model object.
        """
        model_path = ModelType.get_path(model_name)
        if model_path is None:
            raise ValueError(f"Model path not found for model type: {model_name}")
        elif model_name in [ModelType.ID_DETECTOR_MODEL, ModelType.ID_PARTS_DETECTOR_MODEL,
                            ModelType.ID_NUMBER_DETECTOR_MODEL, ModelType.MRZ_DETECTOR_MODEL, 
                            ModelType.DOCUMENT_VERIFIER_MODEL]:
            return YoloModel(model_path=model_path, device=device)
        elif model_name in  []:
            return OnnxModel(model_path=model_path, device=device)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")