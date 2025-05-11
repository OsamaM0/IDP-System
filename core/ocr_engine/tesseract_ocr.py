from typing import Dict, Any, List, Optional
import cv2
import pytesseract
import numpy as np
from pathlib import Path

from utils.image_preprocessor import ImagePreprocessor
from .base_ocr_engine import BoundingBox, OCREngineInterface, OCRResult
from .ocr_engine_enums import OCREngineType, OCRLanguage
from config.config import get_settings

class TesseractOCREngine(OCREngineInterface):
    """Implementation of OCR Engine Interface for Tesseract"""
    
    def __init__(self, languages: List[OCRLanguage], **kwargs):
        # Remove extra kwargs not used by TesseractOCREngine
        kwargs.pop("confidence_threshold", None)
        settings = get_settings()
        self._validate_tesseract_setup(settings)
        self.supported_languages = [
            lang.value if lang != OCRLanguage.ARABIC else 'ara'
            for lang in languages
        ]
                                    
        
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
        
        self.tessdata_dir_config = f'--tessdata-dir {settings.TESSERACT_DIR} --oem 3 --psm 6'
        
    def _validate_tesseract_setup(self, settings) -> None:
        """Validate Tesseract installation and configuration"""
        if not Path(settings.TESSERACT_PATH).exists():
            raise FileNotFoundError(f"Tesseract executable not found at: {settings.TESSERACT_PATH}")
        if not Path(settings.TESSERACT_DIR).exists():
            raise FileNotFoundError(f"Tesseract data directory not found at: {settings.TESSERACT_DIR}")
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        def get_skew_angle(image):
            # Invert the image to make lines white on a black background
            inverted_image = cv2.bitwise_not(image)
            
            # Detect edges in the image
            edges = cv2.Canny(inverted_image, 50, 150, apertureSize=3)
            
            # Use Hough Line Transform to detect lines in the image
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            angles = []

            # Iterate over detected lines and calculate their angles
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    angles.append(angle)

            # Calculate the median angle to minimize the effect of noise
            if len(angles) > 0:
                median_angle = np.median(angles)
                return median_angle
            else:
                return 0  # If no angle is detected, return 0

        def rotate_image(image, angle):
            # Get the image center
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)

            # Calculate the rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Perform the rotation
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        # Get the skew angle
        # skew_angle = get_skew_angle(image)

        # Rotate the image to correct skew
        # corrected_image = rotate_image(image, skew_angle)
        # cv2.imwrite("corrected_image.png", corrected_image)  # Save the corrected image for debugging
        corrected_image = image
        return corrected_image
    
    def get_text(self, image: np.ndarray, lang: Optional[str] = None) -> List[str]:
        """Extract text from image"""
        try:
            image = self._preprocess_image(image)
            text = pytesseract.image_to_string(
                image, 
                lang=lang or self.supported_languages[0],
                config=self.tessdata_dir_config
            )
            return "\n".join([line.strip() for line in text.split('\n') if line.strip()])
        except pytesseract.TesseractError as e:
            raise RuntimeError(f"Tesseract OCR failed: {str(e)}")
    
    
    
    def get_bboxes(self, image: np.ndarray, lang: Optional[str] = None) -> List[BoundingBox]:
        """Get bounding boxes for text regions"""
        try:
            image = self._preprocess_image(image)
            height = image.shape[0]
            boxes = pytesseract.image_to_boxes(
                image,
                lang=lang or self.supported_languages[0],
                # config=self.tessdata_dir_config
            )
            
            result = []
            for box in boxes.splitlines():
                parts = box.split()
                if len(parts) >= 4:
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    result.append(BoundingBox(
                        x1=x1,
                        y1=height - y2,
                        x2=x2,
                        y2=height - y1
                    ))
            return result
        except pytesseract.TesseractError as e:
            raise RuntimeError(f"Tesseract OCR failed: {str(e)}")

    def get_text_with_bbox(self, image: np.ndarray, lang: Optional[str] = None) -> List[OCRResult]:
        """Get text with corresponding bounding boxes"""
        try:
            image = self._preprocess_image(image)
            data = pytesseract.image_to_data(
                image,
                lang=lang or self.supported_languages[0],
                # config=self.tessdata_dir_config,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            for i in range(len(data['text'])):
                if not data['text'][i].strip():
                    continue
                    
                bbox = BoundingBox(
                    x1=data['left'][i],
                    y1=data['top'][i],
                    x2=data['left'][i] + data['width'][i],
                    y2=data['top'][i] + data['height'][i]
                )
                                
                results.append(OCRResult(
                    text=data['text'][i],
                    bbox=bbox,
                    metadata={
                        'block_num': data['block_num'][i],
                        'par_num': data['par_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i],
                        'engine': 'tesseract'
                    }
                ))
            return results
        except pytesseract.TesseractError as e:
            raise RuntimeError(f"Tesseract OCR failed: {str(e)}")

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()
    
    # Added to satisfy abstract method requirement:
    def get_text_with_bounding_boxes(self, image: np.ndarray) -> List[OCRResult]:
        return self.get_text_with_bbox(image)