from core.roi.nid_roi import NID_ROI
from core.roi.passport_roi import MRZ_ROI
from core.roi.base_roi import BaseROI
from core.roi.general_roi import GENERAL_ROI
from core.document_type.document_type_enums import DocumentType
class ROIExtractorFactory:
    """Factory class to create OCR engine objects based on the OCR engine type."""

    @staticmethod
    def create_roi_extractor(roi_type: DocumentType, image: str) -> BaseROI:
        """Creates an ROI extractor object based on the type of ROI extractor.

        Args:
            roi_type (str): The ROI extractor type (e.g., "NID", "Passport", "Tax").

        Raises:
            ValueError: If the ROI extractor type is not supported.

        Returns:
            BaseROI: An ROI extractor object.
        """
        if roi_type == DocumentType.NIDF or roi_type == DocumentType.NIDB:
            return NID_ROI(image=image)
        elif roi_type == DocumentType.PASSPORT:
            return MRZ_ROI(image=image)
        else:
            return GENERAL_ROI(image=image)
