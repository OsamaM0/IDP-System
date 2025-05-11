from core.roi.nid_roi import NID_ROI
from core.roi.passport_roi import MRZ_ROI
from core.roi.base_roi import BaseROI
from core.document_type.document_type_enums import DocumentType
from core.template_parser.general_parser import GeneralParser
from core.template_parser.id_card_parser import IDCardParser
from core.template_parser.mrz_parser import MRZParser
class ParserFactory:
    """Factory class to create OCR engine objects based on the OCR engine type."""

    @staticmethod
    def create_parser(parser_type: DocumentType) -> BaseROI:
        """Creates an parser object based on the type.

        Args:
            parser_type (str): The parser extractor type (e.g., "NID", "Passport", "Tax").

        Raises:
            ValueError: If the parser extractor type is not supported.

        Returns:
            BaseROI: An parser extractor object.
        """
        if parser_type == DocumentType.NIDF or parser_type == DocumentType.NIDB:
            return IDCardParser()
        elif parser_type == DocumentType.PASSPORT:
            return MRZParser()
        else:
            return GeneralParser()

