from typing import List
import numpy as np
from PIL import Image
import easyocr
from .base_ocr_engine import OCREngineInterface, BoundingBox, OCRResult
from .ocr_engine_enums import OCRLanguage, OCREngineType

class EasyOCREngine(OCREngineInterface):
    """OCR Engine Interface implementation using EasyOCR."""
    
    def __init__(self, languages: List[OCRLanguage], use_gpu: bool = False):
        # Initialize EasyOCR Reader with selected languages
        lang_codes = [lang.value for lang in languages]  # e.g., ['en', 'ar']
        self.reader = easyocr.Reader(lang_codes, gpu=use_gpu)  # loads PyTorch models :contentReference[oaicite:2]{index=2}
        self.supported_languages = lang_codes.copy()


    def _convert_to_pil(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    def _convert_bbox(self, points: List[List[float]]) -> BoundingBox:
        # EasyOCR returns 4 corner points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        xs, ys = zip(*points)
        return BoundingBox(x1=int(min(xs)), y1=int(min(ys)),
                           x2=int(max(xs)), y2=int(max(ys)))

    def get_text(self, image: np.ndarray) -> List[str]:
        """Extract high-confidence text lines."""
        results = self.reader.readtext(image)  # [(bbox, text, conf), ...] :contentReference[oaicite:3]{index=3}
        print(results)
        return [text for _, text, conf in results if conf > 0.50]

    def get_bboxes(self, image: np.ndarray) -> List[BoundingBox]:
        """Return bounding boxes for all detected text."""
        results = self.reader.readtext(image)
        return [self._convert_bbox(bbox) for bbox, _, _ in results]

    def get_text_with_bbox(self, image: np.ndarray) -> List[OCRResult]:
        """Combine text and bbox into OCRResult objects."""
        results = []
        for bbox, text, conf in self.reader.readtext(image):
            if conf > 0.80:
                results.append(OCRResult(
                    text=text,
                    bbox=self._convert_bbox(bbox),
                    metadata={"engine": "easyocr", "confidence": conf}
                ))
        return results

    def get_supported_languages(self) -> List[str]:
        return self.supported_languages.copy()
