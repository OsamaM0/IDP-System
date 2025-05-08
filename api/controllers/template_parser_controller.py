from typing import Dict
import numpy as np
from core.document_type.document_type_enums import DocumentType
from core.template_parser.roi_parser import Region, TemplateParser

class TemplateParserController:
    """
    Controller for handling image parsing requests.
    """
    @staticmethod
    def parse_image(img: np.ndarray, parser_type: str) -> Dict[str, Dict[str, Region]]:
        """
        Parses the given image using the specified parser.

        Args:
            img (ndarray): The image to be parsed.
            parser_type (str): The type of parser to use. 
                               Must be one of ["NID", "Tax", "Passport"].

        Returns:
            dict: A dictionary containing the parsing results.
        """
        # Load the precomputed base embeddings (replace with real embeddings)
        valid_documents = DocumentType.get_all_values()
        if parser_type not in valid_documents:
            raise ValueError(f"Invalid parser type: {parser_type}. "
                             f"Valid types are: {valid_documents}")
        parser = TemplateParser(parser_type)
        results = parser.image_hot_region(img)
        return results