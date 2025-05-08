from typing import List, Optional
from pydantic import BaseModel

from core.document_type.document_type_enums import DocumentType
from core.ocr_engine.base_ocr_engine import OCRResult

class OCRResponse(BaseModel):
    results: List[OCRResult]
    document_type: DocumentType
    language: str
    processing_time_ms: float
