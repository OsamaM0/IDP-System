from fastapi import APIRouter, UploadFile, HTTPException
from utils.image_utils import read_image
from core.ocr_engine.ocr_engine_enums import OCRLanguage 
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.factories.ocr_engine_factory import OCREngineFactory
from fastapi import status
import logging

# Add logger to track errors
logger = logging.getLogger(__name__)


router = APIRouter()

@router.post("/ocr")
async def ocr(file: UploadFile, ocr_engine_type: OCREngineType, language: OCRLanguage):
    """OCR the uploaded image file using the specified preprocessor type.
    
    Args:
        file (UploadFile): Request file to preprocess
        preprocessor_type (DocumentType): Preprocessor type to use
        
    Raises:
        HTTPException: Returns an HTTPException if the file is not uploaded
        HTTPException: Returns an HTTPException if the preprocessor_type is not valid
        
    Returns:
        StreamingResponse: Returns the preprocessed image as a StreamingResponse
    """
    if not file:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail="No file uploaded")
    valid_documents = OCREngineType
    if ocr_engine_type not in valid_documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Invalid preprocessor_type: {ocr_engine_type}. Valid options are: {valid_documents}")
    if language not in OCRLanguage:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Invalid language: {language}. Valid options are: {OCRLanguage.ocr_languages()}")
    
    # Read the image and preprocess it
    image_bytes = await file.read()
    image = read_image(image_bytes=image_bytes)
    
    # Initialize the OCR engine with the specified type and language
    # Use a try-except block to handle pickle serialization errors
    try:
        ocr_engine = OCREngineFactory.create_ocr_engine(ocr_engine_type, languages=[language])
    except TypeError as e:
        if "cannot pickle" in str(e):
            logger.warning(f"Pickling error with OCR engine: {e}. Using direct initialization without caching.")
            # Get the engine class directly from the factory's registry
            engine_class = OCREngineFactory._engines[ocr_engine_type]["class"]
            # Get engine config
            config = OCREngineFactory._get_engine_config(ocr_engine_type, [language])
            # Create engine instance directly without caching
            ocr_engine = engine_class(languages=[language], **config)
            logger.info(f"Created non-cached {ocr_engine_type.value} OCR engine")
        else:
            # If it's a different TypeError, re-raise it
            raise
    except EOFError as eof_err:
        logger.warning(f"Encountered cache EOFError: {eof_err}. Clearing cache and retrying.")
        OCREngineFactory.clear_cache()  # clear cache to remove stale entries
        ocr_engine = OCREngineFactory.create_ocr_engine(ocr_engine_type, languages=[language])
    
    # Perform OCR on the image
    text = ocr_engine.get_text(image=image)

    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to extract text")
        
    return [{"extracted_text": text}]