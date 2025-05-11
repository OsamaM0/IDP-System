from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import cv2
import os
import time
from contextlib import contextmanager

from api.controllers.roi_controller import ROIController
from core.document_type.document_verification_controls import DocumentVerificationControls
from core.document_type.document_classifier import DocumentVerification
from core.factories.parser_factory import ParserFactory
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.factories.ocr_engine_factory import OCREngineFactory
from core.document_type.document_type_enums import DocumentType
from utils.image_preprocessor import ImagePreprocessor
from utils.image_upscaler import ImageUpscaler
from utils.logging_utils import logger, log_execution_time, exception_handler
from core.exceptions import (
    DocumentClassificationException,
    PreprocessingException,
    DocumentParsingException
)
from core.cache.cache_manager import cached

# Use dependency injection for better testability
verifier = DocumentVerification(threshold=0.65)
verifier.load_model()


class IDPController:
    """
    Controller for Intelligent Document Processing operations.
    Orchestrates the document processing pipeline including:
    - Document type detection
    - Region of interest extraction
    - OCR processing
    - Template parsing
    """

    @staticmethod
    @log_execution_time
    @exception_handler
    def apply_idp(
        image: np.ndarray,
        ocr_engine_type: OCREngineType, 
        language: OCRLanguage,
        doc_type: Optional[DocumentType] = None
    ) -> Dict[str, Any]:
        """
        Apply the complete IDP pipeline to process a document image.
        
        Args:
            image: The document image as a numpy array
            ocr_engine_type: The OCR engine to use
            language: The language for OCR processing
            doc_type: Optional document type. If not provided, will attempt to detect.
            
        Returns:
            Dictionary containing extracted document information
            
        Raises:
            DocumentClassificationException: If document type cannot be determined
            OCREngineException: If OCR processing fails
            PreprocessingException: If image preprocessing fails
            DocumentParsingException: If document parsing fails
        """
        start_time = time.time()
        
        # Validate inputs
        IDPController._validate_inputs(ocr_engine_type, language)
        
        # Detect document type if not provided
        if doc_type is None:
            logger.info("No document type provided, attempting to detect automatically")
            doc_type = IDPController._detect_document_type(image)
            
        # Validate the detected or provided document type
        if doc_type not in DocumentType.get_all_values():
            raise DocumentClassificationException(
                f"Invalid document type: {doc_type}. Valid types: {DocumentType.get_all_values()}"
            )
        
        logger.info(f"Processing document of type: {doc_type.value}")
        
        # Extract regions of interest
        try:
            extracted_roi = ROIController(doc_type, image).extract_roi_from_image()
            if not extracted_roi:
                raise PreprocessingException(
                    str(doc_type),
                    "Failed to extract regions of interest"
                )
        except Exception as e:
            logger.error(f"ROI extraction failed: {str(e)}")
            raise PreprocessingException(str(doc_type), f"ROI extraction failed: {str(e)}")
            
        # Process each region of interest
        extracted_data = {}
        for region_name, region_image in extracted_roi.items():
            try:
                # Process and enhance region image
                processed_image = IDPController._preprocess_region_image(region_image)
                
                # Get appropriate OCR engine based on region type
                ocr_engine = IDPController._get_ocr_engine_for_region(region_name, ocr_engine_type, language)
                
                # Perform OCR
                ocr_result = ocr_engine.get_text(processed_image)

                extracted_data[region_name] = ocr_result
                
                # Save region for debugging if in debug mode
                IDPController._save_debug_image(region_name, processed_image)
                
                logger.info(f"Successfully processed region: {region_name}")
            except Exception as e:
                logger.error(f"Failed to process region {region_name}: {str(e)}")
                extracted_data[region_name] = f"ERROR: {str(e)}"
        
        # Parse the extracted data using the appropriate parser
        try:
            parser = ParserFactory().create_parser(doc_type)
            parsed_data = parser.parse(extracted_data)
            
            # Add processing metadata
            processing_time = time.time() - start_time
            parsed_data = IDPController._add_metadata(
                parsed_data, 
                doc_type, 
                ocr_engine_type, 
                language, 
                processing_time
            )
            
            return parsed_data
        except Exception as e:
            logger.error(f"Document parsing failed: {str(e)}")
            raise DocumentParsingException(str(doc_type), f"Parsing failed: {str(e)}")

    @staticmethod
    def _validate_inputs(ocr_engine_type: OCREngineType, language: OCRLanguage) -> None:
        """Validate OCR engine type and language inputs."""
        if ocr_engine_type not in OCREngineType:
            raise ValueError(
                f"Invalid OCR engine type: {ocr_engine_type}. "
                f"Valid types are: {[engine.value for engine in OCREngineType]}"
            )
        
        if language not in OCRLanguage:
            raise ValueError(
                f"Invalid language: {language}. "
                f"Valid languages are: {[lang.value for lang in OCRLanguage]}"
            )
        
        logger.info(f"Using OCR engine: {ocr_engine_type.value} with language: {language.value}")

    @staticmethod
    def _detect_document_type(image: np.ndarray) -> DocumentType:
        """
        Detect the document type from an image.
        
        Args:
            image: The document image
            
        Returns:
            Detected DocumentType
            
        Raises:
            DocumentClassificationException: If document cannot be classified
        """
        verifier_results = verifier.verify_document(image)
        
        if not verifier_results:
            raise DocumentClassificationException(
                "Document type could not be determined. Please provide a valid document type."
            )
        
        # Use the first detected document type
        detected_doc = verifier_results[0]
        doc_type = detected_doc.class_name
        
        logger.info(f"Detected document type: {doc_type.value} with confidence: {detected_doc.confidence:.2f}")
        
        # Crop image to the detected document if we have valid bounds
        if detected_doc.bbox and len(detected_doc.bbox) == 4:
            bbox = detected_doc.bbox
            # Ensure bbox values are valid before cropping
            if all(isinstance(coord, (int, float)) for coord in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                try:
                    # Add a small margin to avoid cutting off content
                    margin = 10
                    x1 = max(0, bbox[0] - margin)
                    y1 = max(0, bbox[1] - margin)
                    x2 = min(image.shape[1], bbox[2] + margin)
                    y2 = min(image.shape[0], bbox[3] + margin)
                    
                    image = image[int(y1):int(y2), int(x1):int(x2)]
                except Exception as e:
                    logger.warning(f"Failed to crop image to document bounds: {str(e)}")
        
        return doc_type

    @staticmethod
    def _preprocess_region_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess a region image for OCR.
        
        Args:
            image: The region image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale for better OCR results
        # if len(image.shape) > 2 and image.shape[2] > 1:
        #     image = ImagePreprocessor.convert_to_grayscale(image)
        
        # Upscale small images for better OCR results
        if image.shape[0] < 1100 or image.shape[1] < 1100:
            scale = 1100 // max(image.shape[0], image.shape[1])
            scale = max(1, min(scale, 4))  # Limit scale between 1 and 4
            
            try:
                # Try multiple upscaling methods and use the best one
                methods = [
                    (ImageUpscaler.pixelMapping, "pixel mapping"),
                    (ImageUpscaler.bilinearInterpolation, "bilinear"),
                    (lambda img, s: cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), "cubic")
                ]
                
                for method, name in methods:
                    try:
                        upscaled = method(image, scale)
                        if upscaled is not None and upscaled.size > 0:
                            logger.debug(f"Upscaled image using {name} method by factor {scale}")
                            image = upscaled
                            break
                    except Exception as e:
                        logger.debug(f"Upscaling with {name} failed: {str(e)}")
            except Exception as e:
                logger.warning(f"All upscaling methods failed: {str(e)}")
        
        # Convert back to RGB for OCR engines that require it
        if len(image.shape) == 2:
            image = ImagePreprocessor.convert_to_rgb(image)
            
        # Apply additional enhancements if needed
        # image = ImagePreprocessor.enhance_contrast(image)

        return image

    @staticmethod
    def _get_ocr_engine_for_region(
        region_name: str, 
        default_engine: OCREngineType, 
        default_language: OCRLanguage
    ) -> 'OCREngineInterface':
        """
        Get the appropriate OCR engine for a specific region.
        
        Args:
            region_name: The name of the region
            default_engine: Default OCR engine to use
            default_language: Default OCR language to use
            
        Returns:
            OCR engine instance
        """
        # Special handling for specific regions
        if region_name == "mrz":
            return OCREngineFactory.create_ocr_engine(
                OCREngineType.TESSERACT, 
                languages=[OCRLanguage.MRZ]
            )
        elif region_name in ["nid", "dob", "nid_back", "expiry", "id_number"]:
            return OCREngineFactory.create_ocr_engine(
                OCREngineType.TESSERACT, 
                languages=[OCRLanguage.ARABIC_NUMBER]
            )
        elif region_name in ["name_arabic", "address_arabic"]:
            return OCREngineFactory.create_ocr_engine(
                OCREngineType.PADDLE, 
                languages=[OCRLanguage.ARABIC]
            )
        else:
            # Use the default engine
            return OCREngineFactory.create_ocr_engine(
                default_engine, 
                languages=[default_language]
            )

    @staticmethod
    def _save_debug_image(region_name: str, image: np.ndarray) -> None:
        """Save region image for debugging purposes if debug mode is enabled."""
        import os
        debug_dir = "./res/results" #os.environ.get("IDP_DEBUG_DIR")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            try:
                cv2.imwrite(os.path.join(debug_dir, f"region_{region_name}.png"), image)
            except Exception as e:
                logger.debug(f"Failed to save debug image: {str(e)}")

    @staticmethod
    def _add_metadata(
        data: Dict[str, Any], 
        doc_type: DocumentType, 
        ocr_engine: OCREngineType, 
        language: OCRLanguage,
        processing_time: float
    ) -> Dict[str, Any]:
        """Add metadata to the parsed document data."""
        # Create a new dictionary to avoid modifying the original
        result = data.copy() if isinstance(data, dict) else {"parsed_data": data}
        
        # Add metadata
        result["metadata"] = {
            "document_type": doc_type.value,
            "ocr_engine": ocr_engine.value,
            "language": language.value,
            "processing_time_seconds": round(processing_time, 3),
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        return result

    @staticmethod
    def _process_with_fallback(primary_engine: OCREngineType, fallback_engine: OCREngineType, 
                             image: np.ndarray, language: OCRLanguage) -> Tuple[str, float]:
        """
        Process text with primary OCR engine and fall back to secondary if needed.
        
        Args:
            primary_engine: First OCR engine to try
            fallback_engine: Backup OCR engine to use if primary fails
            image: Image to process
            language: Language to use for OCR
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        try:
            # Try primary engine
            primary_ocr = OCREngineFactory.create_ocr_engine(primary_engine, languages=[language])
            primary_results = primary_ocr.get_text_with_bounding_boxes(image)
            
            if primary_results and len(primary_results) > 0:
                # Calculate average confidence
                confidence = sum(r.confidence for r in primary_results) / len(primary_results)
                
                # Check if confidence is acceptable
                if confidence >= 0.5:
                    text = " ".join([r.text for r in primary_results])
                    return text, confidence
                else:
                    logger.info(f"Low confidence ({confidence:.2f}) with {primary_engine.value}, trying fallback")
            
            # If primary failed or had low confidence, try fallback
            fallback_ocr = OCREngineFactory.create_ocr_engine(fallback_engine, languages=[language])
            fallback_results = fallback_ocr.get_text_with_bounding_boxes(image)
            
            if fallback_results and len(fallback_results) > 0:
                confidence = sum(r.confidence for r in fallback_results) / len(fallback_results)
                text = " ".join([r.text for r in fallback_results])
                return text, confidence
            
            # If all engines failed, return empty result
            return "", 0.0
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return "", 0.0
