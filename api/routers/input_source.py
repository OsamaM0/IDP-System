from fastapi import  HTTPException
from pydantic import BaseModel, Field
from core.input.file_input import FileInput
from core.input.url_input import UrlInput
from core.input.scanner_input import ScannerInput
from core.factories.input_source_factory import InputSourceFactory
from api.schemas.input_source import InputDataSchema, ResponseSchema
from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger('uvicorn.error')

router = APIRouter(
    # prefix="/api/v1",
    # tags=["api_v1", "Input Source"],
)

# Factory Integration Endpoint
@router.post("/process-input", response_model=ResponseSchema)
def process_input(data: InputDataSchema):
    """
    API endpoint to process input data and return the input type.

    Args:
        data (InputDataSchema): Input data provided by the client.

    Returns:
        ResponseSchema: A response indicating the input type.
    """
    try:
        input_source = InputSourceFactory.create_input_source(data.input_data)
        
        if isinstance(input_source, UrlInput):
            return ResponseSchema(input_type="URL", message=f"URL input detected: {data.input_data}")
        elif isinstance(input_source, FileInput):
            return ResponseSchema(input_type="File", message=f"File input detected: {data.input_data}")
        elif isinstance(input_source, ScannerInput):
            return ResponseSchema(input_type="Scanner", message="Scanner input detected.")
        else:
            raise HTTPException(status_code=400, detail="Unknown input type detected.")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
