from core.preprocessor.nid_preprocessor import NIDPreprocessor
from core.preprocessor.tax_preprocessor import TaxPreprocessor
from core.preprocessor.passport_preprocessor import PassportPreprocessor
from core.preprocessor.base_preprocessor import BasePreprocessor
from core.document_type.document_type_enums import DocumentType
import os

class PreprocessorFactory:
    """Factory class to create input source objects based on the input type."""

    @staticmethod
    def create_preprocessor(preprocessor: str) -> BasePreprocessor:
        """
        Creates a Preprocessor object based on the type of preprocessor.
        
        Args:
            preprocessor: The preprocessor type (e.g., "Nid", "Tax", "Passport").
        
        Returns:
            A Preprocessor object or raises an exception if the preprocessor type is not supported.
        """

        if preprocessor == DocumentType.NIDF.value:
            return NIDPreprocessor()
        elif preprocessor == DocumentType.TAX.value:
            return TaxPreprocessor()
        elif preprocessor == DocumentType.PASSPORT.value:
            return PassportPreprocessor()
        else:
            raise ValueError(f"Unsupported preprocessor type: {preprocessor}")