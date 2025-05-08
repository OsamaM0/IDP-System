import numpy as np
from utils.image_preprocessor import ImagePreprocessor
from core.factories.model_factory import ModelFactory
from core.ai_model.model_type_enums import ModelType
from utils.image_utils import preprocess_image, rotate_card
from utils.image_preprocessor import ImagePreprocessor

class NID_ROI:
    def __init__(self, image: np.ndarray = None):
        self.image = image
        
        if image is not None and not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

    def roi_extraction(self) -> dict[str, list[dict[str, list[int]]]]:
        """
        Extract the region of interest from the image.
        """
        # Load the model to extract the region of interest (ID card)
        model = ModelFactory.create_model(model_name=ModelType.ID_DETECTOR_MODEL, device="cpu")
        model.load_model(1)
        results = model.predict(self.image)
        
        cropped_image = None
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                print(f"[INFO] Bounding box: ({x1}, {y1}, {x2}, {y2})")
                cropped_image = rotate_card(self.image, x1, y1, x2, y2)
        preprocessed_rotated_cropped_image = ImagePreprocessor.resize(cropped_image, scale=5)
        
        # Extract ID parts using the preprocessed image
        model = ModelFactory.create_model(model_name=ModelType.ID_PARTS_DETECTOR_MODEL, device="cpu")
        model.load_model(11)
        results = model.predict(preprocessed_rotated_cropped_image)
        output = {"image": preprocessed_rotated_cropped_image,
                  "detected_parts": []}
        
        print(results)
        for result in results:
            output_path = 'd2.jpg'
            result.save(output_path)
            # Append to output the detected part of image with it's class name

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                print(f"[INFO] Detected {class_name} with bbox: {x1, y1, x2, y2}")
                
                output["detected_parts"].append({class_name: [x1, y1, x2, y2]})
        return output