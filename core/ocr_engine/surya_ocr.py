from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from .base_ocr_engine import OCREngineInterface, BoundingBox, OCRResult
from .ocr_engine_enums import OCRLanguage, OCREngineType

class SuryaOCREngine(OCREngineInterface):
    """Implementation of OCR Engine Interface for Surya OCR model"""
    
    def __init__(self, languages: List[OCRLanguage] ):
        """Initialize Surya models and processors"""
        self.det_processor = load_det_processor()
        self.det_model = load_det_model()
        self.rec_model = load_rec_model()
        self.rec_processor = load_rec_processor()
        self.supported_languages = [lang.value for lang in languages] + ['en']
        
    def _convert_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        return Image.fromarray(image)
    
    def _convert_bbox(self, bbox: List[float]) -> BoundingBox:
        """Convert Surya bbox format to interface BoundingBox"""
        return BoundingBox(
            x1=int(bbox[0]),
            y1=int(bbox[1]),
            x2=int(bbox[2]),
            y2=int(bbox[3]),
        )

    def get_text(self, image: np.ndarray) -> List[str]:
        """Extract text from image"""
        pil_image = self._convert_to_pil(image)
        predictions = run_ocr(
            [pil_image],
            [self.supported_languages],
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor
        )
        
        if not predictions or not predictions[0].text_lines:
            return []
        return [line.text for line in predictions[0].text_lines if line.confidence > 0.9]

    def get_bboxes(self, image: np.ndarray) -> List[BoundingBox]:
        """Get bounding boxes for text regions"""
        pil_image = self._convert_to_pil(image)
        predictions = run_ocr(
            [pil_image],
            [self.supported_languages],
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor
        )
        
        if not predictions or not predictions[0].text_lines:
            return []
            
        return [self._convert_bbox(line.bbox) for line in predictions[0].text_lines]

    def get_text_with_bbox(self, image: np.ndarray) -> List[OCRResult]:
        """Get text with corresponding bounding boxes"""
        pil_image = self._convert_to_pil(image)
        predictions = run_ocr(
            [pil_image],
            [self.supported_languages],
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor
        )
        
        if not predictions or not predictions[0].text_lines:
            return []
            
        results = []
        for line in predictions[0].text_lines:
            print(line)
            if line.confidence > 90:
                results.append(OCRResult(
                    text=line.text,
                    bbox=self._convert_bbox(line.bbox),
                    metadata={"engine": "surya"}
                ))
            else:
                print("CAN'T DETECT: ", line)
        return results

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()