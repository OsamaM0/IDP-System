from typing import Dict, Any, List, Union
import numpy as np
from core.document_type.document_type_enums import DocumentType
from core.template_parser.id_card_parser import IDCardParser
from core.template_parser.mrz_parser import MRZParser
from core.factories.parser_factory import ParserFactory

class Region:
    """
    Represents a region of interest in a document.
    """
    def __init__(self, x1: int, y1: int, x2: int, y2: int, label: str = None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the region to a dictionary.
        """
        return {
            "x1": self.x1, 
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "label": self.label
        }


class TemplateParser:
    """
    Parser for document templates that identifies regions of interest and extracts data.
    """
    def __init__(self, parser_type: Union[str, DocumentType]):
        """
        Initialize with the parser type.
        
        Args:
            parser_type: Type of document to parse (e.g., "NIDF", "PASSPORT")
        """
        self.parser_type = parser_type
        self.parser = ParserFactory.create_parser(parser_type)

    def image_hot_region(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Identify regions of interest in the image and extract data.
        
        Args:
            image: The document image as numpy array
            
        Returns:
            Dictionary containing extracted regions and data
        """
        # Extract regions based on document type
        regions = {}
        
        # Basic placeholder implementation - would be replaced with actual region extraction
        if isinstance(self.parser, IDCardParser):
            # Process ID card
            extracted_data = self.parser.parse({"nid": "Sample ID", "demo": "ذكر مسلم"})
            regions = {key: {"value": value} for key, value in extracted_data.items()}
            
        elif isinstance(self.parser, MRZParser):
            # Process MRZ documents (passport, etc)
            extracted_data = self.parser.parse({"mrz": "Sample MRZ Text"})
            regions = {key: {"value": value} for key, value in extracted_data.items()}
            
        else:
            # Generic approach
            extracted_data = {}
        
        return {
            "document_type": self.parser_type,
            "regions": regions,
            "status": "success" if regions else "no_regions_found"
        }
