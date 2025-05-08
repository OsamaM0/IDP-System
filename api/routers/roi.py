import io
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ..controllers.roi_controller import ROIController
from utils.image_utils import numpy_to_bytes, read_image
from core.document_type.document_type_enums import DocumentType 
from fastapi import status 
router = APIRouter()

@router.post("/roi")
async def roi(file: UploadFile, 
              document_type: DocumentType):
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
    if document_type not in valid_documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Invalid preprocessor_type: {document_type}. Valid options are: {valid_documents}")
        
    # Read the image and preprocess it
    image_bytes = await file.read()
    image = read_image(image_bytes=image_bytes)
    
    roi_controller = ROIController(document_type, image)
    if not roi_controller:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create preprocessor controller")
    result = roi_controller.extract_roi()
    if not result:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to extract ROI")

    # Convert the result to json serializable format
    result_json = [dict(zip(item.keys(), item.values())) for item in result.get("detected_parts", [])]
    return result_json