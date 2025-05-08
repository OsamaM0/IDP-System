from typing import Dict
import numpy as np
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.template_parser.roi_parser import Region
from core.factories.ocr_engine_factory import OCREngineFactory
from utils.image_preprocessor import ImagePreprocessor


class OCRController:

    @staticmethod
    def extract_image_text(image: np.ndarray,ocr_engine_type: OCREngineType, 
                           language: OCRLanguage) -> Dict[str, Dict[str, Region]]:
        
        # Load the precomputed base embeddings (replace with real embeddings)
        if ocr_engine_type not in OCREngineType:
            raise ValueError(f"Invalid OCR engine type: {ocr_engine_type}. "
                             f"Valid types are: {[engine.value for engine in OCREngineType]}")
        
        if language not in OCRLanguage:
            raise ValueError(f"Invalid language: {language}. "
                             f"Valid languages are: {[lang.value for lang in OCRLanguage]}")
            
        # Initialize the OCR engine with the specified type and language
        ocr_engine = OCREngineFactory.create_ocr_engine(ocr_engine_type, languages=[language])
        
        # Perform OCR on the image
        text = ocr_engine.get_text(image=image)
        
        return text