import io
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from utils.image_utils import numpy_to_bytes, read_image
from ..controllers.preprocessor_controller import PreprocessorController
from api.schemas.preprocessor import PreprocessorResponseSchema
from core.document_type.document_type_enums import DocumentType 
from fastapi import status 
from core.document_type.document_type_enums import DocumentType

router = APIRouter()

@router.post("/preprocessing")
async def preprocessing(file: UploadFile, 
                        preprocessor_type: DocumentType):
    """Preprocess the uploaded image file using the specified preprocessor type.

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

    valid_documents = DocumentType.get_all_values()
    if preprocessor_type not in valid_documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Invalid preprocessor_type: {preprocessor_type}. Valid options are: {valid_documents}")
        
    # Read the image and preprocess it
    image_bytes = await file.read()
    image = read_image(image_bytes=image_bytes)
    
    preprocessor_controller = PreprocessorController(preprocessor_type, image)
    result = preprocessor_controller.preprocess()
    
    image_base64 = numpy_to_bytes(result)
    
    return StreamingResponse(io.BytesIO(image_base64), media_type="image/png")
    