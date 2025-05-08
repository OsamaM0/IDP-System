from typing import List
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from .base_ocr_engine import OCREngineInterface, BoundingBox, OCRResult
from .ocr_engine_enums import OCRLanguage, OCREngineType

class PaddleOCREngine(OCREngineInterface):
    """Implementation of OCR Engine Interface for Paddle OCR model"""
    
    def __init__(self, languages: List[OCRLanguage]):
        """
        Initialize PaddleOCR with the given language.
        
        Parameters:
        - lang (str): Language code for OCR. Default is English ('en').
        """
        # Initializes PaddleOCR with angle classification enabled (if needed)
        self.languages = languages
        self.ocr = PaddleOCR(lang=languages[0].value, use_angle_cls=True)
    
    def _convert_bbox(self, bbox: List[List[float]]) -> BoundingBox:
        """
        Convert PaddleOCR bbox format to interface BoundingBox.
        
        PaddleOCR returns bbox as a list of four points like:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        We create a box using the minimum x and y and the maximum x and y.
        """
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        return BoundingBox(
            x1=int(min(xs)),
            y1=int(min(ys)),
            x2=int(max(xs)),
            y2=int(max(ys))
        )
    
    def get_text(self, image: np.ndarray) -> List[str]:
        """
        Extract text from the image using PaddleOCR.
        
        Parameters:
        - image (np.ndarray): Input image as a NumPy array.
        
        Returns:
        - List[str]: A list of recognized text strings.
        """
        pil_image = Image.fromarray(image)
        results = self.ocr.ocr(np.array(pil_image), cls=True, det=True)
        print("Results:", results)
        if not results:
            return []
        # results is a list of detected text lines; extract text part from each
        texts = self.sort_and_merge_ocr_results(results[0])
        return texts

    def get_bboxes(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Get bounding boxes for text regions in the image.
        
        Parameters:
        - image (np.ndarray): Input image as a NumPy array.
        
        Returns:
        - List[BoundingBox]: A list of bounding boxes detected.
        """
        pil_image = Image.fromarray(image)
        results = self.ocr.ocr(np.array(pil_image), cls=True)
        if not results:
            return []
        bboxes = [self._convert_bbox(line[0]) for line in results]
        return bboxes

    def get_text_with_bbox(self, image: np.ndarray) -> List[OCRResult]:
        """
        Get text along with corresponding bounding boxes from the image.
        
        Parameters:
        - image (np.ndarray): Input image as a NumPy array.
        
        Returns:
        - List[OCRResult]: A list of OCRResult containing both text and bbox.
        """
        pil_image = Image.fromarray(image)
        results = self.ocr.ocr(np.array(pil_image), cls=True)
        if not results:
            return []
        ocr_results = []
        for line in results[0]:
            bbox = self._convert_bbox(line[0])
            text = line[1][0]
            ocr_results.append(OCRResult(text=text, bbox=bbox, metadata={"engine": "paddle_ocr"}))
        return ocr_results

    def get_supported_languages(self) -> List[str]:
        """
        Return a list of supported languages.
        """
        return self.languages.copy()
    
    def _fix_arabic_text(self, text):
        if not text:
            return text
        # Check if there are Arabic characters in the text
        if any('\u0600' <= c <= '\u06FF' for c in text):
            # Reshape the Arabic text
            s_text = [t[::-1].strip() for t in text.split(" ")]
            s_text.reverse()
            text = " ".join(s_text)
        return text
                
    def sort_and_merge_ocr_results(self, results, y_threshold=50):
        centers_and_texts = []

        for result in results:
            if not result:
                continue
            box, (text, confidence) = result
            # Calculate center of bounding box
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            center_x = sum(x_coords) / 4
            center_y = sum(y_coords) / 4
            centers_and_texts.append(((center_x, center_y), text.strip()))

        # Sort all by Y to find the median and determine split
        centers_and_texts.sort(key=lambda item: item[0][1])
        all_ys = [pt[0][1] for pt in centers_and_texts]
        if len(all_ys) == 0:
            return ""
        # Use median to decide line split
        mean_y = (max(all_ys) + min(all_ys)) / 2
    
        upper_line = [item for item in centers_and_texts if item[0][1] < mean_y - y_threshold]
        lower_line = [item for item in centers_and_texts if item[0][1] >= mean_y - y_threshold]

        # Sort each line by X
        upper_line.sort(key=lambda item: item[0][0])
        lower_line.sort(key=lambda item: item[0][0])

        print("UPPER LINE: ", upper_line)
        print("LOWER LINE: ", lower_line)
        # Combine text
        upper_text = ' '.join([self._fix_arabic_text(item[1]) for item in upper_line][::-1])
        lower_text = ' '.join([self._fix_arabic_text(item[1]) for item in lower_line][::-1])
        print("UPPER TEXT: ", upper_text)
        print("LOWER TEXT: ", lower_text)
        return "\n".join([lower_text, upper_text])  if upper_text else lower_text
