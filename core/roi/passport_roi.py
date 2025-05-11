import numpy as np
from .base_roi import BaseROI
from core.ai_model.model_type_enums import ModelType
from core.factories.model_factory import ModelFactory 

class MRZ_ROI(BaseROI):
    def __init__(self, image=None):
        self.image = image
        
        if image is not None and not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

    def roi_extraction(self) -> dict[str, list[dict[str, list[int]]]]:
        # Ensure input is a numpy array
        if not isinstance(self.image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        
        # Preprocess the image if needed
        # preprocessor = PassportPreprocessor()
        # self.image = preprocessor.preprocess(self.image)
        
        model = ModelFactory.create_model(model_name=ModelType.MRZ_DETECTOR_MODEL, device="cpu")
        model.load_model()
        results = model.predict(self.image)
        
        output = {"image": self.image, "detected_parts": []}
        
        for result in results:
            output_path = 'mrz2.jpg'
            result.save(output_path)
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                bbox = [int(coord) for coord in bbox]
                output["detected_parts"].append({class_name: bbox})
        
        return output
