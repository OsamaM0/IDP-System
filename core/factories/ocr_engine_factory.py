from typing import List, Optional, Dict, Type, Any, Tuple, Union, cast
import os
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import psutil
import logging

from core.ocr_engine.easy_ocr import EasyOCREngine
from core.ocr_engine.surya_ocr import SuryaOCREngine
from core.ocr_engine.tesseract_ocr import TesseractOCREngine
from core.ocr_engine.paddle_ocr import PaddleOCREngine
from core.ocr_engine.base_ocr_engine import OCREngineInterface
from core.ocr_engine.yolo_ar_num import YoloArabicNumberOCR
from ..ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.exceptions import OCREngineException
from utils.logging_utils import logger, exception_handler
from core.di.interfaces import IOCREngine
from core.cache.cache_manager import cached
from config.config import get_settings


class OCREngineFactory:
    """
    Advanced factory for creating and managing OCR engines based on type and language.
    
    This implementation includes enhanced caching, thread safety, dynamic configuration,
    and optimized engine selection strategies for different document types.
    """
    
    # Registry of available OCR engines with metadata
    _engines: Dict[OCREngineType, Dict[str, Any]] = {
        OCREngineType.EASY_OCR: {
            "class": EasyOCREngine,
            "strengths": ["multilingual", "handwritten"],
            "weaknesses": ["speed", "specialized_formats"]
        },
        OCREngineType.SURYA: {
            "class": SuryaOCREngine,
            "strengths": ["arabic", "specialized_formats"],
            "weaknesses": ["rare_languages"]
        },
        OCREngineType.TESSERACT: {
            "class": TesseractOCREngine,
            "strengths": ["structured_text", "mrz", "tabular"],
            "weaknesses": ["handwritten", "low_resolution"]
        },
        OCREngineType.PADDLE: {
            "class": PaddleOCREngine,
            "strengths": ["general_purpose", "arabic", "natural_scenes"],
            "weaknesses": ["very_small_text"]
        },
        OCREngineType.YOLO_AR_NUM: {
            "class": YoloArabicNumberOCR,
            "strengths": ["arabic_numbers", "detection_accuracy"],
            "weaknesses": ["text_only", "general_purpose"]
        },
    }
    
    # Thread-safe instance cache using a lock
    _engine_instances: Dict[str, OCREngineInterface] = {}
    _lock = threading.RLock()
    
    # Thread pool for parallel operations
    _executor = ThreadPoolExecutor(max_workers=4)
    
    # Preferred engines by document region and language
    REGION_ENGINE_PREFERENCES = {
        "mrz": [OCREngineType.TESSERACT, OCREngineType.PADDLE],
        "arabic_text": [OCREngineType.PADDLE, OCREngineType.SURYA],
        "id_number": [OCREngineType.TESSERACT, OCREngineType.YOLO_AR_NUM],
        "table": [OCREngineType.TESSERACT, OCREngineType.PADDLE],
        "handwritten": [OCREngineType.EASY_OCR, OCREngineType.PADDLE]
    }
    
    # Language-specific engine preferences
    LANGUAGE_ENGINE_PREFERENCES = {
        OCRLanguage.ARABIC: [OCREngineType.PADDLE, OCREngineType.SURYA],
        OCRLanguage.ARABIC_NUMBER: [OCREngineType.YOLO_AR_NUM, OCREngineType.TESSERACT],
        OCRLanguage.MRZ: [OCREngineType.TESSERACT],
        OCRLanguage.ENGLISH: [OCREngineType.PADDLE, OCREngineType.SURYA, OCREngineType.TESSERACT, OCREngineType.EASY_OCR],
        OCRLanguage.CHINESE: [OCREngineType.PADDLE],
        OCRLanguage.JAPANESE: [OCREngineType.PADDLE],
        # Add more language preferences as needed
    }

    @classmethod
    def manage_cache_memory(cls, max_memory_percent: float = 80.0) -> None:
        """
        Monitors memory usage and clears cache if system memory usage exceeds threshold.
        
        Args:
            max_memory_percent: Maximum memory usage percentage before clearing cache
        """
        try:
            memory_info = psutil.virtual_memory()
            if memory_info.percent > max_memory_percent:
                logger.warning(f"Memory usage at {memory_info.percent}%, clearing OCR engine cache")
                cls.clear_cache()
                return True
            return False
        except Exception as e:
            logger.error(f"Error monitoring memory: {str(e)}")
            return False

    @classmethod
    @exception_handler
    @cached(ttl=3600)  # Cache for 1 hour
    def create_ocr_engine(
        cls, 
        ocr_engine_type: OCREngineType, 
        languages: List[OCRLanguage],
        max_retries: int = 3,
        **kwargs
    ) -> OCREngineInterface:
        """
        Create or retrieve a cached OCR engine instance with optimized configuration.
        
        Args:
            ocr_engine_type: Type of OCR engine to create
            languages: Languages the engine should support
            max_retries: Maximum number of retries on transient errors
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured OCR engine instance
            
        Raises:
            OCREngineException: If engine creation fails
        """
        retry_count = 0
        cache_key = cls._generate_cache_key(ocr_engine_type, languages, kwargs)
        
        while retry_count < max_retries:
            try:
                # Check if we need to manage memory first
                cls.manage_cache_memory()
                
                # Thread-safe access to engine instances
                with cls._lock:
                    # Return cached instance if available
                    if cache_key in cls._engine_instances:
                        logger.debug(f"Using cached OCR engine: {ocr_engine_type.value}")
                        return cls._engine_instances[cache_key]
                
                # Validate engine type
                if ocr_engine_type not in cls._engines:
                    error_msg = f"Unsupported OCR engine type: {ocr_engine_type.value}"
                    logger.error(error_msg)
                    raise OCREngineException(error_msg, str(ocr_engine_type))
                
                try:
                    # Get the engine class
                    engine_class = cls._engines[ocr_engine_type]["class"]
                    
                    # Apply engine-specific configuration with smart defaults
                    config_kwargs = cls._get_engine_config(ocr_engine_type, languages)
                    merged_kwargs = {**config_kwargs, **kwargs}
                    
                    # Initialize the engine
                    logger.info(f"Creating {ocr_engine_type.value} OCR engine with languages: {[lang.value for lang in languages]}")
                    
                    engine = engine_class(languages=languages, **merged_kwargs)
                    
                    # Cache the instance
                    with cls._lock:
                        cls._engine_instances[cache_key] = engine
                    
                    logger.info(f"Successfully created OCR engine: {ocr_engine_type.value}")
                    return engine
                    
                except (EOFError, ConnectionError, TimeoutError) as transient_err:
                    # These are transient errors that might benefit from retry
                    retry_count += 1
                    error_msg = f"Transient error creating OCR engine (attempt {retry_count}/{max_retries}): {str(transient_err)}"
                    logger.warning(error_msg)
                    
                    if retry_count >= max_retries:
                        raise OCREngineException(error_msg, str(ocr_engine_type)) from transient_err
                        
                    # Clear cache and wait before retry
                    cls.clear_cache()
                    time.sleep(0.5 * retry_count)  # Exponential backoff
                    
                except Exception as e:
                    error_msg = f"Failed to create OCR engine {ocr_engine_type.value}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise OCREngineException(error_msg, str(ocr_engine_type)) from e

            except Exception as e:
                error_msg = f"Failed to create OCR engine {ocr_engine_type.value}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise OCREngineException(error_msg, str(ocr_engine_type)) from e
    
    @classmethod
    def _generate_cache_key(cls, engine_type: OCREngineType, languages: List[OCRLanguage], kwargs: Dict[str, Any]) -> str:
        """Generate a unique cache key for an OCR engine configuration."""
        # Sort language values for consistent keys regardless of order
        lang_str = "-".join(sorted([lang.value for lang in languages]))
        
        # Include only relevant kwargs in the key
        relevant_kwargs = {
            k: str(v) for k, v in kwargs.items() 
            if k in ["confidence_threshold", "use_gpu", "use_angle_cls"]
        }
        kwargs_str = "-".join(f"{k}={v}" for k, v in sorted(relevant_kwargs.items()))
        
        # Create the complete key
        key_parts = [engine_type.value, lang_str]
        if kwargs_str:
            key_parts.append(kwargs_str)
            
        return "_".join(key_parts)
    
    @classmethod
    def _get_engine_config(cls, engine_type: OCREngineType, languages: List[OCRLanguage] = None) -> Dict[str, Any]:
        """Get optimized configuration parameters for a specific OCR engine."""
        settings = get_settings()
        
        # Common settings for all engines
        common_config = {
            "confidence_threshold": settings.OCR_CONFIDENCE_THRESHOLD
        }
        
        # Check for GPU availability
        use_gpu = getattr(settings, "USE_GPU", False)
        
        # Engine-specific configurations with optimizations
        engine_configs = {
            OCREngineType.PADDLE: {
                "use_angle_cls": True,
                "use_gpu": use_gpu,
                "show_log": False,
                "lang": "ar",  # Default language
                "det_limit_side_len": 2560,  # Optimize for larger documents
                "det_limit_type": "max",
                "det_db_thresh": 0.3,
                "det_db_box_thresh": 0.5,
            },
            OCREngineType.TESSERACT: {
                "config": "--psm 6 --oem 3",
                "timeout": 30,  # Prevent hanging on complex images
                "lang": "ara+eng",  # Default language combo
            },
            OCREngineType.EASY_OCR: {
                "gpu": use_gpu,
                "model_storage_directory": os.path.join(os.getcwd(), "models", "easyocr"),
                "detector": True,
                "recognizer": True,
            },
            OCREngineType.SURYA: {
                "batch_size": 4,
                "cache_dir": os.path.join(os.getcwd(), "models", "surya"),
            },
            OCREngineType.YOLO_AR_NUM: {
                "confidence_threshold": 0.4,  # More lenient for Arabic numbers
                "model_path": getattr(settings, "YOLO_AR_NUM_MODEL_PATH", os.path.join(os.getcwd(), "models", "yolo_ar_num")),
            }
        }
        
        # Get engine-specific config with fallback to empty dict
        specific_config = engine_configs.get(engine_type, {})
        
        # Language-specific adjustments if languages are provided
        if languages and len(languages) > 0:
            primary_language = languages[0]
            
            # Adjust Tesseract configuration based on language
            if engine_type == OCREngineType.TESSERACT:
                if primary_language == OCRLanguage.MRZ:
                    specific_config["config"] = "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                elif primary_language == OCRLanguage.ARABIC_NUMBER:
                    specific_config["config"] = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
            
            # Adjust PaddleOCR based on language
            elif engine_type == OCREngineType.PADDLE:
                lang_map = {
                    OCRLanguage.ARABIC: "ar",
                    OCRLanguage.ENGLISH: "en",
                    OCRLanguage.CHINESE: "ch",
                    OCRLanguage.JAPANESE: "japan",
                    OCRLanguage.KOREAN: "korean",
                    OCRLanguage.MRZ: "en",
                    OCRLanguage.ARABIC_NUMBER: "ar",
                }
                specific_config["lang"] = lang_map.get(primary_language, "en")
        
        # Merge configurations with engine-specific taking precedence
        return {**common_config, **specific_config}
    
    @classmethod
    def get_supported_engines(cls) -> List[str]:
        """Get a list of all supported OCR engine types."""
        return [engine_type.value for engine_type in cls._engines.keys()]
    
    @classmethod
    def get_recommended_engine(cls, language: OCRLanguage, region_type: str = None) -> OCREngineType:
        """
        Get the optimal OCR engine for a specific language and region type.
        
        Args:
            language: Target language for OCR
            region_type: Optional region type (e.g., 'mrz', 'id_number')
            
        Returns:
            The recommended OCR engine type
        """
        # Check region-specific preferences first if a region type is provided
        if region_type and region_type in cls.REGION_ENGINE_PREFERENCES:
            preferred_engines = cls.REGION_ENGINE_PREFERENCES[region_type]
            # Return the first available preferred engine
            for engine in preferred_engines:
                if engine in cls._engines:
                    return engine
        
        # Check language-specific preferences
        if language in cls.LANGUAGE_ENGINE_PREFERENCES:
            preferred_engines = cls.LANGUAGE_ENGINE_PREFERENCES[language]
            # Return the first available preferred engine
            for engine in preferred_engines:
                if engine in cls._engines:
                    return engine
        
        # Default to PaddleOCR as a general-purpose engine
        return OCREngineType.PADDLE
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached OCR engine instances to free memory."""
        with cls._lock:
            cls._engine_instances.clear()
        logger.debug("OCR engine cache cleared")
    
    @classmethod
    async def preload_engines(cls, languages: List[OCRLanguage] = None) -> None:
        """
        Proactively load common OCR engines to improve first-request latency.
        
        Args:
            languages: Optional list of languages to preload, defaults to Arabic and English
        """
        if languages is None:
            languages = [OCRLanguage.ARABIC, OCRLanguage.ENGLISH]
            
        engines_to_preload = [OCREngineType.PADDLE, OCREngineType.TESSERACT]
        
        try:
            # Use thread pool to load engines concurrently
            futures = []
            for engine_type in engines_to_preload:
                for language in languages:
                    futures.append(
                        cls._executor.submit(
                            cls.create_ocr_engine, 
                            engine_type, 
                            [language]
                        )
                    )
            
            # Wait for all engines to load
            for future in futures:
                future.result()
                
            logger.info(f"Preloaded {len(futures)} OCR engine configurations")
        except Exception as e:
            logger.warning(f"Engine preloading failed: {str(e)}")
