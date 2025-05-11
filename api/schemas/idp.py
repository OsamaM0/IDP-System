from enum import Enum
from typing import Dict, List, Optional, Union, Any
from fastapi import File, UploadFile
from pydantic import BaseModel, Field, validator
from core.document_type.document_type_enums import DocumentType
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage

class InputSourceType(str, Enum):
    AUTO = "auto"
    FILE = "file" 
    URL = "url"
    SCANNER = "scanner"
    BYTES = "bytes"

class IDPRequestSchema(BaseModel):
    """Schema for IDP processing requests"""
    input_data: Optional[str] = Field(None, description="Input data (URL, file path, or scanner command)")
    input_type: InputSourceType = Field(default=InputSourceType.AUTO, description="Type of input source")
    ocr_engine_type: OCREngineType = Field(default=OCREngineType.PADDLE, description="OCR engine to use")
    language: OCRLanguage = Field(default=OCRLanguage.ARABIC, description="Language for OCR processing")
    doc_type: Optional[DocumentType] = Field(None, description="Document type (if known)")
    file: Optional[UploadFile] = File(None)

    @validator('input_data')
    def validate_input_data(cls, v, values):
        input_type = values.get('input_type')
        if input_type != InputSourceType.BYTES and not v:
            raise ValueError("input_data is required unless input_type is 'bytes'")
        return v

class ExtractedField(BaseModel):
    """Represents an extracted field from a document"""
    name: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bounding_box: Optional[List[int]] = None

class IDPResponseSchema(BaseModel):
    """Schema for IDP processing response"""
    document_type: DocumentType
    language: str
    processing_time_ms: float
    extracted_fields: List[ExtractedField]
    raw_ocr_text: Optional[str] = None
    image_hash: Optional[str] = None
