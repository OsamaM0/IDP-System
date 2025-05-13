from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import time
import torch
import re
from paddleocr import PaddleOCR

from utils.image_preprocessor import ImagePreprocessor
from .base_ocr_engine import OCREngineInterface, BoundingBox, OCRResult
from .ocr_engine_enums import OCRLanguage, OCREngineType
from utils.logging_utils import logger, log_execution_time, exception_handler
from core.exceptions import OCREngineException
from core.cache.cache_manager import cached
from concurrent.futures import ThreadPoolExecutor

class PaddleOCREngine(OCREngineInterface):
    """Implementation of OCR Engine Interface for Paddle OCR model with enhanced text processing"""
    
    def __init__(self, languages: List[OCRLanguage], **kwargs):
        """
        Initialize PaddleOCR model with specific languages and configuration.
        
        Args:
            languages: List of languages to initialize the model with
            **kwargs: Additional configuration parameters including:
                confidence_threshold: Minimum confidence score for text detection
                use_angle_cls: Whether to detect text orientation
                use_gpu: Whether to use GPU acceleration if available
                det_db_thresh: Detection binary threshold
                det_db_box_thresh: Detection box threshold
                rec_model_dir: Path to recognition model
                det_model_dir: Path to detection model
        """
        super().__init__(languages, **kwargs)
        self.supported_languages = [lang.value for lang in languages]
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.use_gpu = kwargs.get('use_gpu', torch.cuda.is_available())
        self.use_angle_cls = kwargs.get('use_angle_cls', True)
        
        # Store languages for language-specific processing
        self.languages = languages
        self.primary_language = languages[0] if languages else OCRLanguage.ENGLISH
        
        # Configure GPU usage
        self.use_gpu = torch.cuda.is_available() and kwargs.get('use_gpu', True)
        logger.info(f"Using GPU for PaddleOCR: {self.use_gpu}")
        
        # Configure angle classification for rotated text
        self.use_angle_cls = kwargs.get('use_angle_cls', True)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=kwargs.get('max_workers', 4))
        
        # Initialize language-specific processing rules
        self._init_language_rules()
        
        # Initialize PaddleOCR with appropriate language
        lang = self._map_language(languages[0])
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=lang,
                use_gpu=self.use_gpu,
                show_log=False,
                det_db_thresh=kwargs.get('det_db_thresh', 0.3),
                det_db_box_thresh=kwargs.get('det_db_box_thresh', 0.5),
                rec_model_dir=kwargs.get('rec_model_dir', None),
                det_model_dir=kwargs.get('det_model_dir', None),
                det_limit_side_len=kwargs.get('det_limit_side_len', 2560),
                det_limit_type=kwargs.get('det_limit_type', 'max')
            )
            logger.info(f"Initialized PaddleOCR engine with language: {lang}, GPU: {self.use_gpu}")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise OCREngineException(f"Failed to initialize PaddleOCR: {str(e)}", "PaddleOCR")
    
    def _init_language_rules(self):
        """Initialize language-specific processing rules"""
        # Map of language to specific regex patterns or functions for post-processing
        self.language_processors = {
            OCRLanguage.ARABIC: {
                'normalize': self._normalize_arabic,
                'fix_common_errors': self._fix_arabic_errors,
                'direction': 'rtl',
                'char_replacements': {
                    # Common Arabic character confusions
                    'ا': 'ا',  # Alef variations
                    'أ': 'ا',
                    'إ': 'ا',
                    'آ': 'ا',
                    'ي': 'ي',  # Ya variations
                    'ى': 'ي',
                    'ئ': 'ي',
                }
            },
            OCRLanguage.ENGLISH: {
                'normalize': self._normalize_latin,
                'fix_common_errors': self._fix_english_errors,
                'direction': 'ltr',
                'char_replacements': {
                    # Common English OCR errors
                    '0': 'O',
                    'l': 'I',
                    '1': 'I',
                    '5': 'S',
                    '$': 'S',
                    '8': 'B',
                }
            },
            OCRLanguage.ARABIC_NUMBER: {
                'normalize': self._normalize_numbers,
                'fix_common_errors': lambda x: x,  # No specific fixes
                'direction': 'ltr',
                'char_replacements': {
                    # Ensure digits are parsed correctly
                    'o': '0',
                    'O': '0',
                    'l': '1',
                    'I': '1',
                    'B': '8',
                }
            },
            OCRLanguage.MRZ: {
                'normalize': self._normalize_mrz,
                'fix_common_errors': self._fix_mrz_errors,
                'direction': 'ltr',
                'char_replacements': {
                    # MRZ specific confusions
                    '0': 'O',
                    '1': 'I',
                    '8': 'B',
                    '2': 'Z',
                    '5': 'S',
                    ' ': '<',
                }
            }
        }
    
    @staticmethod
    def _map_language(language: OCRLanguage) -> str:
        """
        Map OCRLanguage enum to PaddleOCR language code.
        
        Args:
            language: Language enum
            
        Returns:
            Language code string for PaddleOCR
        """
        # Define mapping from OCRLanguage enum to PaddleOCR language codes
        language_map = {
            OCRLanguage.ARABIC: "ar",
            OCRLanguage.ENGLISH: "en",
            OCRLanguage.CHINESE: "ch",
            OCRLanguage.JAPANESE: "japan",
            OCRLanguage.KOREAN: "korean",
            OCRLanguage.GERMAN: "german",
            OCRLanguage.FRENCH: "french",
            OCRLanguage.ITALIAN: "it",
            OCRLanguage.SPANISH: "es",
            OCRLanguage.RUSSIAN: "ru",
            OCRLanguage.PORTUGUESE: "pt",
            OCRLanguage.ARABIC_NUMBER: "ar",
            OCRLanguage.MRZ: "en"
        }
        
        # Default to English if language not found in map
        return language_map.get(language, "en")

    @log_execution_time
    @exception_handler
    # @cached(ttl=0)  # Cache results for 5 minutes
    def get_text(self, image: np.ndarray) -> str:
        """
        Extract text from an image using PaddleOCR with enhanced processing.
        
        Args:
            image: Input image as a numpy array
            
        Returns:
            Extracted text as a string
            
        Raises:
            OCREngineException: If OCR processing fails
        """
        try:
            # Apply image preprocessing specific to PaddleOCR
            processed_image = self.preprocess_image(image)           
            # Run OCR on the image
            start_time = time.time()
            result = self.ocr.ocr(processed_image, cls=True)
            ocr_time = time.time() - start_time
            logger.debug(f"PaddleOCR extraction completed in {ocr_time:.2f}s")
            
            # Handle empty results
            if not result or len(result) == 0 or not result[0]:
                logger.warning("No text detected in image")
                return ""
            # Process and merge OCR results with advanced layout analysis
            text = self.sort_and_merge_ocr_results(result[0])
            # Apply post-processing for the primary language
            primary_language = self.languages[0] if self.languages else OCRLanguage.ENGLISH
            # text = self.postprocess_text(text, primary_language)
            
            logger.debug(f"Extracted text length: {len(text)}")
            return text
            
        except Exception as e:
            logger.error(f"PaddleOCR text extraction failed: {str(e)}")
            raise OCREngineException(f"Text extraction failed: {str(e)}", "PaddleOCR")

    @log_execution_time
    @exception_handler
    def get_text_with_bounding_boxes(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text with bounding box information with enhanced confidence handling.
        
        Args:
            image: Input image as a numpy array
            
        Returns:
            List of OCRResult objects containing text and bounding boxes
            
        Raises:
            OCREngineException: If OCR processing fails
        """
        try:
            # Apply image preprocessing
            processed_image = self.preprocess_image(image)
            
            # Run OCR on the image
            start_time = time.time()
            result = self.ocr.ocr(processed_image, cls=True)
            ocr_time = time.time() - start_time
            logger.debug(f"PaddleOCR extraction with boxes completed in {ocr_time:.2f}s")
            
            # Handle empty results
            if not result or len(result) == 0 or not result[0]:
                logger.warning("No text detected in image")
                return []
            
            # Convert PaddleOCR results to standardized OCRResult objects
            ocr_results: List[OCRResult] = []
            primary_language = self.languages[0] if self.languages else OCRLanguage.ENGLISH
            
            for entry in result[0]:
                if not entry or len(entry) != 2:
                    continue
                
                box, (text, confidence) = entry
                
                # Skip results below confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Create bounding box from PaddleOCR polygon format
                if len(box) == 4:  # Ensure we have 4 points for the box
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    
                    bbox = BoundingBox(
                        x1=int(min(x_coords)),
                        y1=int(min(y_coords)),
                        x2=int(max(x_coords)),
                        y2=int(max(y_coords))
                    )
                    
                    # Apply text post-processing based on language
                    processed_text = self.postprocess_text(text, primary_language)
                    
                    # Store additional polygon points for more complex layout analysis
                    polygon = [(int(pt[0]), int(pt[1])) for pt in box]
                    
                    ocr_result = OCRResult(
                        text=processed_text,
                        bbox=bbox,
                        confidence=float(confidence),
                        language=self.detect_text_language(processed_text),
                        polygon=polygon
                    )
                    
                    ocr_results.append(ocr_result)
            
            # Sort results by vertical position for better reading order
            ocr_results.sort(key=lambda x: x.bbox.y1)
            return ocr_results
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction with bounding boxes failed: {str(e)}")
            raise OCREngineException(f"Extraction with bounding boxes failed: {str(e)}", "PaddleOCR")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image before OCR to improve text detection.
        
        Args:
            image: Input image as a numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if image is None or image.size == 0:
            logger.error("Empty image provided to PaddleOCR")
            raise ValueError("Empty image provided")
        
        # # Ensure image is in RGB format
        # if len(image.shape) == 2:  # Grayscale image
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # elif image.shape[2] == 4:  # RGBA image, convert to RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # elif len(image.shape) == 3 and image.shape[2] == 3 and np.max(image) <= 1.0:  
        #     # Normalized image [0,1], convert to [0,255]
        #     image = (image * 255).astype(np.uint8)
        
        # # Apply contrast enhancement for better text detection
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # lab_planes = list(cv2.split(lab))  # Convert tuple to list for mutable assignment
        # lab_planes[0] = clahe.apply(lab_planes[0])
        # lab = cv2.merge(lab_planes)
        # enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # # Apply language-specific preprocessing
        # primary_language = self.languages[0] if self.languages else OCRLanguage.ENGLISH
        
        # # Apply additional enhancements for Arabic if needed
        # if primary_language == OCRLanguage.ARABIC:
        #     # For Arabic text, apply specific enhancements like sharpening
        #     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #     enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
        
        # # For MRZ, optimize with different preprocessing
        # if primary_language == OCRLanguage.MRZ:
        #     # Apply binary threshold
        #     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #     # Convert back to RGB
        #     enhanced_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
                # Expand the image to avoid cutting off text
        enhanced_image = ImagePreprocessor.expand_image_background(image, 10, 1.5) 

        return enhanced_image

    def postprocess_text(self, text: str, language: OCRLanguage = None) -> str:
        """
        Apply text postprocessing based on language to improve OCR results.
        
        Args:
            text: Raw OCR text
            language: Language for specialized processing
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Use the first language if not specified
        if language is None:
            language = self.languages[0] if self.languages else OCRLanguage.ENGLISH
        
        # Apply common fixes
        text = self._fix_common_errors(text)
        
        # Apply language-specific processing if available
        if language in self.language_processors:
            processor = self.language_processors[language]
            
            # Apply normalization
            if 'normalize' in processor:
                text = processor['normalize'](text)
                
            # Apply language-specific error fixes
            if 'fix_common_errors' in processor:
                text = processor['fix_common_errors'](text)
            
        return text.strip()

    def _fix_common_errors(self, text: str) -> str:
        """Fix common OCR errors in the extracted text."""
        if not text:
            return ""
        
        # Remove unwanted characters
        text = text.replace("\u200f", "")  # Remove right-to-left mark
        text = text.replace("\u200e", "")  # Remove left-to-right mark
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control characters
        
        return text

    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text."""
        if not text:
            return ""
        
        # Normalize alef forms
        text = re.sub('[أإآا]', 'ا', text)
        
        # Normalize taa marbouta
        text = re.sub('ة', 'ه', text)
        
        # Normalize yaa
        text = re.sub('[يى]', 'ي', text)
        
        # Remove kashida (tatweel)
        text = re.sub('ـ', '', text)
        
        # Remove Arabic diacritics
        text = re.sub('[\u064B-\u065F]', '', text)
        
        return text

    def _normalize_latin(self, text: str) -> str:
        """Normalize Latin script text."""
        if not text:
            return ""
        
        # Remove accents from characters
        import unicodedata
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
        
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalize numeric text."""
        if not text:
            return ""
        
        # Keep only digits, spaces and some punctuation
        text = ''.join(c for c in text if c.isdigit() or c in ' .,/-')
        
        return text

    def _normalize_mrz(self, text: str) -> str:
        """Normalize MRZ text."""
        if not text:
            return ""
        
        # Convert to uppercase (MRZ is always uppercase)
        text = text.upper()
        
        # Keep only MRZ valid characters (uppercase letters, numbers and <)
        text = ''.join(c for c in text if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
        
        return text

    def _fix_arabic_errors(self, text: str) -> str:
        """Fix common errors in Arabic text."""
        if not text:
            return ""
        
        # Fix common Arabic character confusions
        replacements = {
            '٠': '0',  # Arabic zero to digit zero
            '١': '1',  # Arabic one to digit one
            '٢': '2',  # etc.
            '٣': '3',
            '٤': '4',
            '٥': '5',
            '٦': '6',
            '٧': '7',
            '٨': '8',
            '٩': '9',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text

    def _fix_english_errors(self, text: str) -> str:
        """Fix common errors in English text."""
        if not text:
            return ""
        
        # Fix common OCR confusions in English
        replacements = {
            '0': 'O',  # Only in contexts where it makes sense
            'l': 'I',  # In some contexts
            '|': 'I',
            'rn': 'm',
            'vv': 'w'
        }
        
        # Apply replacements only in appropriate contexts
        for old, new in replacements.items():
            if old in text:
                # Use regex to apply replacements only in word context
                text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
                
        return text

    def _fix_mrz_errors(self, text: str) -> str:
        """Fix common errors in MRZ text."""
        if not text:
            return ""
        
        # Common MRZ confusions
        replacements = {
            '0': 'O',  # In MRZ, 0 appears as O
            '1': 'I',  # Sometimes confused
            '8': 'B',  # Sometimes confused
            '5': 'S',  # Sometimes confused
            ' ': '',   # Remove spaces in MRZ
        }
        
        # Only apply if it looks like MRZ content (contains < character)
        if '<' in text:
            for old, new in replacements.items():
                text = text.replace(old, new)
        
        return text

    def detect_text_language(self, text: str) -> OCRLanguage:
        """
        Detect the language of the extracted text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language enum
        """
        if not text or len(text) < 2:
            return self.languages[0] if self.languages else OCRLanguage.ENGLISH
            
        # Check for Arabic characters
        if re.search(r'[\u0600-\u06FF]', text):
            return OCRLanguage.ARABIC
            
        # Check for mostly numbers
        if sum(c.isdigit() for c in text) / len(text) > 0.7:
            return OCRLanguage.ARABIC_NUMBER
            
        # Check for MRZ pattern (uppercase, digits and <)
        if re.match(r'^[A-Z0-9<]+$', text) and '<' in text:
            return OCRLanguage.MRZ
            
        # Default to English for Latin characters
        if re.search(r'[A-Za-z]', text):
            return OCRLanguage.ENGLISH
            
        # Use the primary configured language as fallback
        return self.languages[0] if self.languages else OCRLanguage.ENGLISH

    def sort_and_merge_ocr_results(self, results, y_threshold=50) -> str:
        """
        Sort and merge OCR results into coherent text with advanced layout analysis.
        
        Args:
            results: OCR results from PaddleOCR
            y_threshold: Vertical threshold for grouping text lines
            
        Returns:
            Merged text with proper reading order
        """
        if not results:
            return ""
            
        centers_and_texts = []

        for result in results:
            if not result:
                continue
                
            box, (text, confidence) = result
            
            # Skip low confidence results
            if confidence < self.confidence_threshold:
                continue
                
            # Calculate center of bounding box
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            center_x = sum(x_coords) / 4
            center_y = sum(y_coords) / 4
            
            # Calculate width and height for better grouping decisions
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            centers_and_texts.append({
                'center': (center_x, center_y),
                'text': text.strip(),
                'confidence': confidence,
                'width': width,
                'height': height,
                'box': box
            })

        if not centers_and_texts:
            return ""

        # Adaptive y_threshold calculation based on text density and font size
        if len(centers_and_texts) > 1:
            # Sort by y-coordinate
            sorted_by_y = sorted(centers_and_texts, key=lambda item: item['center'][1])
            
            # Get median line height
            heights = [item['height'] for item in centers_and_texts]
            median_height = sorted(heights)[len(heights)//2]
            
            # Adjust threshold based on median height
            y_threshold = max(median_height * 0.7, 10)

        # Group into lines by Y coordinate
        centers_and_texts.sort(key=lambda item: item['center'][1])  # Sort by Y coordinate
        
        # Group into lines
        current_line_y = centers_and_texts[0]['center'][1]
        lines = []
        current_line = []
        
        for item in centers_and_texts:
            center_x, center_y = item['center']
            
            # If y is significantly different, start a new line
            if abs(center_y - current_line_y) > y_threshold:
                if current_line:  # Skip empty lines
                    lines.append(sorted(current_line, key=lambda i: i['center'][0]))  # Sort by X within line
                current_line = [item]
                current_line_y = center_y
            else:
                current_line.append(item)
                # Update current_line_y to average of the line items for better accuracy
                line_y_values = [i['center'][1] for i in current_line]
                current_line_y = sum(line_y_values) / len(line_y_values)
                
        # Add the last line
        if current_line:
            lines.append(sorted(current_line, key=lambda i: i['center'][0]))
            
        # Determine primary language direction for text rendering
        primary_language = self.languages[0] if self.languages else OCRLanguage.ENGLISH
        default_direction = 'rtl' if primary_language == OCRLanguage.ARABIC else 'ltr'
        
        # Generate text with appropriate reading order and line breaks
        text_lines = []
        for line in lines:
            # Determine line direction by checking Arabic character ratio
            arabic_char_count = sum(len(re.findall(r'[\u0600-\u06FF]', item['text'])) for item in line)
            total_char_count = sum(len(item['text']) for item in line)
            
            # If more than 30% Arabic characters, treat as RTL
            is_rtl = (arabic_char_count / total_char_count > 0.3) if total_char_count else False
            
            if is_rtl or default_direction == 'rtl':
                # For RTL languages like Arabic, reverse the order
                line_text = ' '.join(self._fix_arabic_text(item['text']) for item in line[::-1])
            else:
                # For LTR languages
                line_text = ' '.join(item['text'] for item in line)
            
            text_lines.append(line_text)
            
        # Join lines with newlines
        return '\n'.join(text_lines)
        
    def _fix_arabic_text(self, text: str) -> str:
        """Fix Arabic text direction and common issues."""
        if not text:
            return ""
            
        # Remove non-Arabic/non-numeric characters if it's an Arabic string
        if any('\u0600' <= c <= '\u06FF' for c in text):
            # Keep Arabic characters, numbers, and punctuation
            result = ''.join(c for c in text if 
                      '\u0600' <= c <= '\u06FF' or  # Arabic
                      c.isdigit() or                # Numbers
                      c in ' .,!?-:;()[]{}/')       # Common punctuation
            # Reverse the order of words for Arabic "Handeling Error in Arabic from PaddleOCR in RTL"
            s_text = [t[::-1].strip() for t in result.split(" ")]
            s_text.reverse()
            text = " ".join(s_text)
            return text
        
        return text
