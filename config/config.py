import os
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import yaml
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings with validation and environment variable support.
    
    Settings can be overridden by environment variables with the prefix IDP_.
    For example, IDP_API_KEY would override API_KEY.
    """
    # API settings
    API_KEY: Optional[str] = Field(None, description="API key for authentication")
    DEBUG: bool = Field(False, description="Debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    
    # Application info
    app_name: str = Field("OCR System", description="Application name")
    app_version: str = Field("0.1", description="Application version")
    
    # Tesseract settings
    TESSERACT_PATH: str = Field("C:/Tesseract-OCR/tesseract.exe", description="Path to Tesseract executable")
    TESSERACT_DIR: str = Field("C:/Tesseract-OCR/tessdata", description="Path to Tesseract data directory")
    
    # Model paths - both uppercase and lowercase versions for backward compatibility
    ID_DETECTOR_MODEL_PATH: str = Field("assets/models/id_detector.pt", description="Path to ID detector model")
    MRZ_DETECTOR_MODEL_PATH: str = Field("assets/models/passport-mrz_detector.pt", description="Path to MRZ detector model")
    ID_PARTS_DETECTOR_MODEL_PATH: str = Field("assets/models/id_parts_detector.pt", description="Path to ID parts detector model")
    ID_NUMBER_DETECTOR_MODEL_PATH: str = Field("assets/models/id_number_detector.pt", description="Path to ID number detector model")
    DOCUMENT_CLASSIFIER_MODEL_PATH: str = Field("assets/models/document_classifier_model.pt", description="Path to document classifier model")
    
    # Lowercase versions for backward compatibility
    id_detector_model_path: Optional[str] = None
    mrz_detector_model_path: Optional[str] = None
    id_parts_detector_model_path: Optional[str] = None
    id_number_detector_model_path: Optional[str] = None
    document_classifier_model_path: Optional[str] = None
    
    # OCR settings
    DEFAULT_OCR_ENGINE: str = Field("paddle", description="Default OCR engine")
    DEFAULT_LANGUAGE: str = Field("ar", description="Default language for OCR")
    OCR_CONFIDENCE_THRESHOLD: float = Field(0.5, description="OCR confidence threshold")
    
    # Performance settings
    ENABLE_CACHING: bool = Field(False, description="Enable caching for expensive operations")
    CACHE_TTL: int = Field(3600, description="Cache time-to-live in seconds")
    BATCH_SIZE: int = Field(8, description="Batch size for processing")
    
    # Security settings
    RATE_LIMIT_ENABLED: bool = Field(True, description="Enable rate limiting")
    RATE_LIMIT: int = Field(100, description="Maximum requests per minute")
    
    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate that the log level is a valid one."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator('LOG_LEVEL')
    def validate_non_empty_string(cls, v, info):
        """Validate that required string fields are not empty."""
        if v is None or (isinstance(v, str) and len(v.strip()) == 0):
            raise ValueError(f"{info.field_name} cannot be empty")
        return v
    
    @field_validator('ID_DETECTOR_MODEL_PATH', 'MRZ_DETECTOR_MODEL_PATH', 
              'ID_PARTS_DETECTOR_MODEL_PATH', 'ID_NUMBER_DETECTOR_MODEL_PATH', 
              'DOCUMENT_CLASSIFIER_MODEL_PATH')
    def validate_file_exists(cls, v):
        """Validate that model paths exist."""
        if not os.path.exists(v):
            # Just warn instead of raising exception to allow for container environments
            import logging
            logging.warning(f"Model path does not exist: {v}")
        return v

    class Config:
        env_prefix = "IDP_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance to avoid reloading settings on every call.
    
    Returns:
        Settings: Application settings
    """
    return Settings()