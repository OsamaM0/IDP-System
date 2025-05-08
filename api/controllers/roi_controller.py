from core.factories.roi_factory import ROIExtractorFactory
import numpy as np

class ROIController:
    def __init__(self, roi_type: str, image: np.ndarray):
        self.roi_type = roi_type
        self.image = image
    
    def extract_roi(self) -> dict[str, list[dict[str, list[int]]]]:
        """
        Extract the region of interest (ROI) from the image using the specified ROI type.
        
        Returns:
            list[dict[str, list[int]]]: A list of dictionaries containing the detected ROIs.
        """
        
        roi_extractor = ROIExtractorFactory.create_roi_extractor(self.roi_type, self.image)
        
        return roi_extractor.roi_extraction()
    
    def extract_roi_from_image(self, ) -> dict[str, np.ndarray]:
        """
        Extract the region of interest (ROI) from the provided image using the specified ROI type.
        
        Args:
            image (np.ndarray): The input image from which to extract the ROI.
        
        Returns:
            list[dict[str, list[int]]]: A list of dictionaries containing the detected ROIs.
        """
        roi_extractor = ROIExtractorFactory.create_roi_extractor(self.roi_type, self.image)
        extracted_roi = roi_extractor.roi_extraction()
        
        extracted_image = extracted_roi.get("image", None)
        
        if extracted_image is None:
            raise ValueError("No image found in the extracted ROI.")
        
        output = {}
        for roi in extracted_roi.get("detected_parts", []):
            for k, v in roi.items():
                print(f"[INFO] Detected {k} with bbox: {v}")
                x1, y1, x2, y2 = v
                roi_image = extracted_image[y1:y2, x1:x2]
                output[k] = roi_image
        return output