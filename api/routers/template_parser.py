import io
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ..controllers.template_parser_controller import TemplateParserController
from core.document_type.document_type_enums import DocumentType
from utils.image_utils import numpy_to_bytes, read_image
from fastapi import status 

router = APIRouter()

@router.post("/template_parser/")
async def verify_document(file: UploadFile, 
                          template_parser_type: DocumentType):
    """ Preprocess the uploaded image file using the specified preprocessor type.

    Args:
        file (UploadFile):  Request file to preprocess
        template_parser_type (DocumentType):  Preprocessor type to use

    Raises:
        HTTPException:  Returns an HTTPException if the file is not uploaded
        HTTPException:  Returns an HTTPException if the preprocessor_type is not valid

    Returns:
        JSONResponse:  Returns the template parser diminsions as a JSONResponse.
    """
    if not file:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail="No file uploaded")
    
    valid_documents = DocumentType.get_all_values()
    if template_parser_type not in valid_documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Invalid preprocessor_type: {template_parser_type}. Valid options are: {valid_documents}")
    
    # Read the image and preprocess it
    image_bytes = await file.read()
    image = read_image(image_bytes=image_bytes)
    
    parser_controller = TemplateParserController.parse_image(image, template_parser_type)
    
    return JSONResponse(content=parser_controller)
