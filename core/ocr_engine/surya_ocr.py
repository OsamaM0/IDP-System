from typing import List
import numpy as np
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from .base_ocr_engine import OCREngineInterface, BoundingBox, OCRResult
from .ocr_engine_enums import OCRLanguage

class SuryaOCREngine(OCREngineInterface):
    """OCR Engine Interface implementation for Surya OCR 0.13.1."""
    def __init__(self, languages: List[OCRLanguage], confidence_threshold=0.5, **kwargs):
        # Initialize predictors
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor()
        self.confidence_threshold = confidence_threshold
        # Supported languages plus English fallback
        self.supported_languages = [lang.value for lang in languages] + ["en"]

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    def _to_bbox(self, bbox: List[float]) -> BoundingBox:
        # Surya outputs [x1, y1, x2, y2]
        return BoundingBox(x1=int(bbox[0]), y1=int(bbox[1]),
                           x2=int(bbox[2]), y2=int(bbox[3]))

    def get_text(self, image: np.ndarray) -> List[str]:
        pil = self._to_pil(image)
        # Perform detection + recognition in one call
        preds = self.rec_predictor([pil], [self.supported_languages], self.det_predictor)
        # print(preds)
        if not preds or not preds[0].text_lines:
            return []
        # Filter by confidence >0.9
        return [
            line.text for line in preds[0].text_lines
            if line.confidence >  self.confidence_threshold
        ]

    def get_bboxes(self, image: np.ndarray) -> List[BoundingBox]:
        pil = self._to_pil(image)
        dets = self.det_predictor([pil])
        if not dets or not dets[0].bboxes:
            return []
        return [self._to_bbox(b) for b in dets[0].bboxes]

    def get_text_with_bounding_boxes(self, image: np.ndarray) -> List[OCRResult]:
        pil = self._to_pil(image)
        preds = self.rec_predictor([pil], [self.supported_languages], self.det_predictor)
        if not preds or not preds[0].text_lines:
            return []
        results = []
        for line in preds[0].text_lines:
            bbox = self._to_bbox(line.bbox)
            conf = line.confidence
            text = line.text if conf > self.confidence_threshold else ""
            results.append(OCRResult(text=text, bbox=bbox,
                                     metadata={"engine": "surya", "confidence": conf}))
        return results

    def get_supported_languages(self) -> List[str]:
        return self.supported_languages.copy()