from fastapi import APIRouter, UploadFile, HTTPException
from utils.image_utils import read_image
from core.ocr_engine.ocr_engine_enums import OCRLanguage 
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.factories.ocr_engine_factory import OCREngineFactory
from fastapi import status 


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
    ocr_engine = OCREngineFactory.create_ocr_engine(ocr_engine_type, languages=[language])
    
    # Perform OCR on the image
    text = ocr_engine.get_text(image=image)

    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to extract text")
        
    return [{"extracted_text": text}]