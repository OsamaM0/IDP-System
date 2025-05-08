import torch
import numpy as np
import logging
from api.schemas.document_verification import VerificationResult
from core.factories.model_factory import ModelFactory
from core.ai_model.model_type_enums import ModelType
from core.document_type.document_type_enums import DocumentType
from core.document_type.document_verification_controls import DocumentVerificationControls

class DocumentVerification:
    def __init__(self, threshold: float = 0.50):
        self.controls = DocumentVerificationControls(threshold=threshold)
    
    def load_model(self):
        """Load the document verification model."""
        return self.controls.load_model()
    
    def verify_document(self, image):
        """Verify the document using the loaded model."""
        return self.controls.verify_document(image)