from typing import List, Tuple, Dict, Any
import numpy as np
from core.ai_model.model_type_enums import ModelType
from core.factories.model_factory import ModelFactory
from core.ocr_engine.base_ocr_engine import OCREngineInterface, BoundingBox, OCRResult  # added imports

class YoloArabicNumberOCR(OCREngineInterface):  # inherit from interface
    """
    OCR Engine for Arabic numbers using YOLO model.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the OCR engine with a YOLO model.
        """
        self.device = device
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model from the specified path."""
        self.model = ModelFactory.create_model(ModelType.ID_NUMBER_DETECTOR_MODEL, device=self.device)
        self.model.load_model(1)
        self.paddle_ocr = None
    
    def detect_numbers(self, image: np.ndarray, confidence: float = 0.25) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect Arabic numbers in an image using YOLO model.
        """
        print("Detecting numbers...")
        print(image.shape)
        results = self.model.predict(image)
        print("Results:")
        print(results)
        # print(results)
        detected_info = []
        detections = []
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detected_info.append((cls, x1))
                detections.append({
                    'digit': cls,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # Sort detections by x-coordinate (left to right)
        detected_info.sort(key=lambda x: x[1])
        number_string = ''.join([str(cls) for cls, _ in detected_info])
        
        return number_string, detections

    def get_text(self, image: np.ndarray) -> List[str]:
        """
        Extract text from image.
        """
        number_string, _ = self.detect_numbers(image)
        print(number_string)
        return [number_string]

    def get_bboxes(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Get bounding boxes for detected numbers.
        """
        _, detections = self.detect_numbers(image)
        return [BoundingBox(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]) for d in detections]

    def get_text_with_bbox(self, image: np.ndarray) -> List[OCRResult]:
        """
        Get text with corresponding bounding boxes.
        """
        _, detections = self.detect_numbers(image)
        return [
            OCRResult(
                text=str(d['digit']),
                bbox=BoundingBox(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]),
                metadata={'confidence': d['confidence']}
            )
            for d in detections
        ]

    def get_supported_languages(self) -> List[str]:
        """
        Return a list of supported languages.
        """
        return ["arabic"]
