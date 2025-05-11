"""
Custom exception classes for the IDP system.
"""
from typing import Optional, Dict, Any, List


class IDPBaseException(Exception):
    """Base exception for all IDP system exceptions."""
    def __init__(self, message: str = "An error occurred in the IDP system"):
        self.message = message
        super().__init__(self.message)


class OCREngineException(IDPBaseException):
    """Raised when there's an issue with OCR processing."""
    def __init__(self, message: str = "OCR processing failed", engine_name: Optional[str] = None):
        self.engine_name = engine_name
        super_message = f"{message} (Engine: {engine_name})" if engine_name else message
        super().__init__(super_message)


class ModelLoadingException(IDPBaseException):
    """Raised when a model cannot be loaded."""
    def __init__(self, model_name: str, message: str = "Failed to load model"):
        self.model_name = model_name
        super().__init__(f"{message}: {model_name}")


class DocumentClassificationException(IDPBaseException):
    """Raised when document classification fails."""
    def __init__(self, message: str = "Failed to classify document"):
        super().__init__(message)


class PreprocessingException(IDPBaseException):
    """Raised when image preprocessing fails."""
    def __init__(self, preprocess_type: str, message: str = "Image preprocessing failed"):
        self.preprocess_type = preprocess_type
        super().__init__(f"{message} for {preprocess_type}")


class InputSourceException(IDPBaseException):
    """Raised when there's an issue with the input source."""
    def __init__(self, source_type: str, message: str = "Input source error"):
        self.source_type = source_type
        super().__init__(f"{message} for {source_type}")


class DocumentParsingException(IDPBaseException):
    """Raised when document parsing fails."""
    def __init__(self, parser_type: str, message: str = "Document parsing failed"):
        self.parser_type = parser_type
        super().__init__(f"{message} for {parser_type}")


class ValidationException(IDPBaseException):
    """Raised when validation of extracted data fails."""
    def __init__(self, field_name: Optional[str] = None, 
                 message: str = "Validation failed", 
                 validation_errors: Optional[List[Dict[str, Any]]] = None):
        self.field_name = field_name
        self.validation_errors = validation_errors or []
        
        if field_name:
            super_message = f"{message} for field '{field_name}'"
        else:
            super_message = message
        super().__init__(super_message)
