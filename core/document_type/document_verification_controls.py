import torch
import numpy as np
import logging
from api.schemas.document_verification import VerificationResult
from core.factories.model_factory import ModelFactory
from core.ai_model.model_type_enums import ModelType
from core.document_type.document_type_enums import DocumentType

class DocumentVerificationControls:
    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = ModelType.DOCUMENT_VERIFIER_MODEL
    
    def load_model(self):
        """Load the document verification model."""
        self.model = ModelFactory.create_model(model_name=self.model_type, device=self.device)
        self.model.load_model()
        return self.model
    
    def verify_document(self, image: np.ndarray) -> list[VerificationResult]:
        """Verify the document using the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model before verification.")
        
        # Perform verification
        results = self.model.predict(image)
        
        output = []
        
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()
                if box.conf[0].item() < self.threshold:
                    continue
                class_id = int(box.cls[0].item())
                class_name = DocumentType.get_document_type_by_value(result.names[class_id])
                bbox = [int(coord) for coord in bbox]
                logging.info(f"[INFO] Detected {class_name} with bbox: {bbox}")
                output.append(VerificationResult(
                    class_name=class_name, bbox=bbox, confidence=box.conf[0].item()
                ))
        if not output:
            logging.info(f"[INFO] No document detected in the image.")
            output.append(VerificationResult(
                class_name=DocumentType.NO_CLASS, bbox=[], confidence=0.0
            ))
        
        return output
