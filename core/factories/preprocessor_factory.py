from core.preprocessor.nid_preprocessor import NIDPreprocessor
from core.preprocessor.tax_preprocessor import TaxPreprocessor
from core.preprocessor.passport_preprocessor import PassportPreprocessor
from core.preprocessor.mrz_preprocessor import MRZPreprocessor
from core.preprocessor.base_preprocessor import BasePreprocessor
from core.document_type.document_type_enums import DocumentType
import os
from utils.logging_utils import logger, exception_handler
from typing import Optional

class PreprocessorFactory:
    """Factory class to create preprocessor objects based on document type."""
    
    @staticmethod
    @exception_handler
    def create_preprocessor(doc_type: DocumentType) -> Optional[BasePreprocessor]:
        """
        Create a preprocessor instance based on document type.
        
        Args:
            doc_type: The type of document to process
            
        Returns:
            An instance of a preprocessor appropriate for the document type
            
        Raises:
            ValueError: If the document type is not supported
        """
        logger.info(f"Creating preprocessor for document type: {doc_type}")
        
        if doc_type == DocumentType.NIDF or doc_type == DocumentType.NIDB:
            return NIDPreprocessor()
        elif doc_type == DocumentType.TAX:
            return TaxPreprocessor()
        elif doc_type == DocumentType.PASSPORT:
            return PassportPreprocessor()
        elif doc_type == DocumentType.MRZ:
            return MRZPreprocessor()
        else:
            logger.warning(f"No specialized preprocessor found for {doc_type}, using default.")
            return None