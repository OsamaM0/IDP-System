# ocr_response.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import base64
from fastapi import Response

from core.document_type.document_type_enums import DocumentType

class VerificationResult(BaseModel):
    class_name: DocumentType = Field(
        description="The name of the detected document type",
        example=DocumentType.NO_CLASS,
    )
    bbox: List[int]
    confidence: float = Field(
        description="The confidence score of the detected document type",
        example=0.95,
    )

class DocumentVerificationResponse(BaseModel):
    image: str = Field(description="Base64-encoded image data")
    verification_results: List[VerificationResult] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_response(self) -> Response:
        """Convert the model to a FastAPI Response with image data and document type header."""
        headers = {}
        if self.verification_results:
            headers["X-Document-Type"] = str(self.verification_results[0].class_name)
            
        return Response(
            content=base64.b64decode(self.image),
            media_type="image/jpeg",
            headers=headers
        )
