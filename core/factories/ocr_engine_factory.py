from typing import List
from core.ocr_engine.easy_ocr import EasyOCREngine
from core.ocr_engine.surya_ocr import SuryaOCREngine
from core.ocr_engine.tesseract_ocr import TesseractOCREngine
from core.ocr_engine.paddle_ocr import PaddleOCREngine
from core.ocr_engine.base_ocr_engine import OCREngineInterface
from core.ocr_engine.yolo_ar_num import YoloArabicNumberOCR
from ..ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage


import os

class OCREngineFactory:
    """Factory class to create OCR engine objects based on the OCR engine type."""

    @staticmethod
    def create_ocr_engine(ocr_engine: OCREngineType, languages: List[OCRLanguage] ) -> OCREngineInterface:
        """ Creates an OCR engine object based on the type of OCR engine.

        Args:
            ocr_engine (str): The OCR engine type (e.g., "Tesseract", "Surya").

        Raises:
            ValueError: If the OCR engine type is not supported.

        Returns:
            OCREngineInterface: An OCR engine object.
        """

        if ocr_engine == OCREngineType.TESSERACT:
            return TesseractOCREngine(languages=languages)
        elif ocr_engine == OCREngineType.SURYA:
            return SuryaOCREngine(languages=languages)
        elif ocr_engine == OCREngineType.PADDLE:
            return PaddleOCREngine(languages=languages)
        elif ocr_engine == OCREngineType.EASY_OCR:
            return EasyOCREngine(languages=languages)
        elif ocr_engine == OCREngineType.YOLO_AR_NUM:
            return YoloArabicNumberOCR()
        else:
            raise ValueError(f"Unsupported preprocessor type: {ocr_engine}")
