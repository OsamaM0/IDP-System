from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import Response
from pydantic import BaseModel, Field
from core.input.file_input import FileInput
from core.input.url_input import UrlInput
from core.input.scanner_input import ScannerInput
from core.factories.input_source_factory import InputSourceFactory
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO

logger = logging.getLogger('uvicorn.error')

router = APIRouter()
scanner = ScannerInput()


class InputDataSchema(BaseModel):
    input_data: str = Field(..., description="Input data (URL, file path, or scanner command)")

@router.post("/process-input")
async def process_input(data: InputDataSchema):
    """
    API endpoint to process input data and return the image.

    Args:
        data (InputDataSchema): Input data provided by the client.
    
    Returns:
        Response: The image data with appropriate content type
    """
    try:
        input_source = InputSourceFactory.create_input_source(data.input_data)
        image_data = input_source.load_image()
        
        # Determine content type based on input source type
        content_type = "image/jpeg"  # Default content type
        if isinstance(input_source, UrlInput):
            # You might want to determine content type from URL or response headers
            content_type = "image/jpeg"  # Adjust based on actual image type
        elif isinstance(input_source, FileInput):
            # You might want to determine content type from file extension
            content_type = "image/jpeg"  # Adjust based on actual image type
        elif isinstance(input_source, ScannerInput):
            content_type = "image/jpeg"  # Adjust based on scanner output format
            
        return Response(
            content=image_data,
            media_type=content_type
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing input")
    
    

@router.get("/scan")
async def scan_document():
    """Endpoint to scan a document."""
    try:
        image_data = await scanner.load_image()
        return StreamingResponse(
            BytesIO(image_data),
            media_type="image/jpeg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scanner/status")
async def get_scanner_status():
    """Get current scanner status."""
    return await scanner.get_status()

@router.on_event("shutdown")
async def shutdown_scanner():
    """Cleanup scanner resources on shutdown."""
    await scanner.shutdown()