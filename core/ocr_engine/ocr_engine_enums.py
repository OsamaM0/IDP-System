from enum import Enum

class OCREngineType(Enum):
    TESSERACT = "tesseract"
    SURYA = "surya"
    PADDLE = "paddle"
    EASY_OCR = "easy_ocr"
    YOLO_AR_NUM = "yolo_ar_num"

class OCRLanguage(Enum):
    ARABIC = "ar"
    ARABIC_NUMBER =  "ara_number"
    FAST = "fast"
    MRZ = "mrz"
    ENGLISH = "en" 
    BENGALI = "bn"
    CHINESE = "zh"
    FRENCH = "fr"
    GERMAN = "de"
    HINDI = "hi"
        
    @classmethod
    def from_code(cls, code: str) -> 'OCRLanguage':
        """
        Finds and returns the language enum based on any code (default or Tesseract).
        :param code: Code to look for.
        :return: Matching OCRLanguage enum.
        """
        for lang in cls:
            if code in lang.value:
                return lang
        raise ValueError(f"No language found for code '{code}'.")
    
    @classmethod
    def ocr_languages(cls) -> list:
        """
        Returns a list of supported OCR languages.
        """
        return [lang.value[0] for lang in cls]
