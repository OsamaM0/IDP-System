from typing import Dict
import numpy as np
from api.controllers.roi_controller import ROIController
from core.document_type.document_verification_controls import DocumentVerificationControls
from core.factories.parser_factory import ParserFactory
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.factories.ocr_engine_factory import OCREngineFactory
from core.document_type.document_type_enums import DocumentType
import cv2

from utils.image_preprocessor import ImagePreprocessor
from utils.image_upscaler import ImageUpscaler 

verifier = DocumentVerificationControls()
verifier.load_model()

class IDPController:
    @staticmethod
    def apply_idp(image: np.ndarray, ocr_engine_type: OCREngineType, 
                  language: OCRLanguage, doc_type: DocumentType = None) -> Dict[str, str]:
        # Validate OCR engine type and language
        if ocr_engine_type not in OCREngineType:
            raise ValueError(f"Invalid OCR engine type: {ocr_engine_type}. "
                             f"Valid types are: {[engine.value for engine in OCREngineType]}")
        
        if language not in OCRLanguage:
            raise ValueError(f"Invalid language: {language}. "
                             f"Valid languages are: {[lang.value for lang in OCRLanguage]}")
        print(f"[INFO] Using OCR engine: {ocr_engine_type.value} with language: {language.value} doc_type: {doc_type}")
        # Detect document type if not provided
        if doc_type is None:
            verifier_results = verifier.verify_document(image)
            if not verifier_results:
                raise ValueError("Document type could not be determined. Please provide a valid document type.")
            
            # Use the first detected document type and crop the image
            detected_doc = verifier_results[0]
            print(f"[INFO] Detected document type: {detected_doc.class_name.value}")
            doc_type = detected_doc.class_name
            bbox = detected_doc.bbox
            image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Crop the image to the detected bounding box
        
        # Validate the detected or provided document type
        if doc_type not in DocumentType.get_all_values():
            raise ValueError(f"Invalid document type: {doc_type}. "
                             f"Valid document types are: {DocumentType.get_all_values()}")
        
        # Get all the regions of interest (ROIs) for the specified document type
        extracted_roi = ROIController(doc_type, image).extract_roi_from_image()
        
        # Perform OCR on the image using the specified OCR engine and language
        extracted_data = {}

        for k, v in extracted_roi.items():
            # Apply preprocessing to the image if needed
            v = ImagePreprocessor.convert_to_grayscale(v)
            
            # Apply upscaling to the image if needed
            height, width, _ = v.shape
            if height < 1100 and width < 1100:
                scale = 1100 // max(height, width)
                v = ImageUpscaler.pixelMapping(v, scale)
                v = ImageUpscaler.bilinearInterpolation(v, scale)
                v = cv2.resize(v, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Perform OCR based on the region type
            if k == "mrz":
                ocr_engine = OCREngineFactory.create_ocr_engine(OCREngineType.TESSERACT, languages=[OCRLanguage.MRZ])
            elif k in ["nid", "dob", "nid_back", "expiry"]:
                ocr_engine = OCREngineFactory.create_ocr_engine(OCREngineType.TESSERACT, languages=[OCRLanguage.ARABIC_NUMBER])
            else:
                ocr_engine = OCREngineFactory.create_ocr_engine(ocr_engine_type, languages=[language])
                if ocr_engine_type == OCREngineType.GOOGLE_VISION:
                    print("[INFO] Using Google Vision OCR engine")
                v = ImagePreprocessor.expand_image_background(v, scale_height=10, scale_width=1.5)
            
            # Perform OCR on the ROI image
            ocr_result = ocr_engine.get_text(v)
            extracted_data[k] = ocr_result
            cv2.imwrite(f"extracted_{k}.png", v)  # Save the extracted ROI image for debugging
            print(f"[INFO] Detected {k} with text: {extracted_data[k]}")
        
        # Apply parser
        parser = ParserFactory().create_parser(doc_type)
        extracted_data = parser.parse(extracted_data)
        
        return extracted_data
