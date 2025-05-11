"""
Interfaces for dependency injection.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.document_type.document_type_enums import DocumentType


class IInputSource(ABC):
    """Interface for input sources."""
    
    @abstractmethod
    async def load_image(self) -> bytes:
        """Load image data from the source."""
        pass


class IPreprocessor(ABC):
    """Interface for image preprocessors."""
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Process the input image and return the result."""
        pass


class IOCREngine(ABC):
    """Interface for OCR engines."""
    
    @abstractmethod
    def get_text(self, image: np.ndarray) -> str:
        """Extract text from an image."""
        pass
    
    @abstractmethod
    def get_text_with_bounding_boxes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text with bounding box information."""
        pass


class IDocumentClassifier(ABC):
    """Interface for document classifiers."""
    
    @abstractmethod
    def classify_document(self, image: np.ndarray) -> DocumentType:
        """Classify a document image."""
        pass


class ITemplateParser(ABC):
    """Interface for document template parsers."""
    
    @abstractmethod
    def parse(self, document_data: Dict[str, str]) -> List[Dict[str, str]]:
        """Parse document data according to template."""
        pass


class IROIExtractor(ABC):
    """Interface for region of interest extraction."""
    
    @abstractmethod
    def extract_roi(self, image: np.ndarray, document_type: DocumentType) -> Dict[str, np.ndarray]:
        """Extract regions of interest from an image."""
        pass


class IDocumentProcessor(ABC):
    """Interface for document processors."""
    
    @abstractmethod
    async def process_document(
        self, 
        image: np.ndarray,
        ocr_engine_type: OCREngineType,
        language: OCRLanguage,
        doc_type: Optional[DocumentType] = None
    ) -> Dict[str, Any]:
        """Process a document and extract structured information."""
        pass
