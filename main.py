import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import time
import uuid
from api.routers import input_source, ocr, document_types, roi, idp, preprocessing, template_parser
from api.middleware.security import SecurityMiddleware
from api.responses.standard_response import create_error_response
from core.exceptions import IDPBaseException
import logging
from utils.logging_utils import logger
from config.config import get_settings
from core.di.container import container
import os

# Get application settings
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title="IDP System API",
    description="Intelligent Document Processing System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware with explicit allowance for documentation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Ensure headers are exposed properly
)

# Add security middleware
app.add_middleware(
    SecurityMiddleware,
    rate_limit_per_minute=settings.RATE_LIMIT,
    api_key=settings.API_KEY
)

# Include API routers
app.include_router(input_source.router, prefix="/api/v1", tags=["Input Source"])
app.include_router(document_types.router, prefix="/api/v1", tags=["Document Types"])
app.include_router(roi.router, prefix="/api/v1", tags=["ROI"])
app.include_router(ocr.router, prefix="/api/v1", tags=["OCR"])
app.include_router(idp.router, prefix="/api/v1", tags=["IDP"])
# Include any other routers that may exist
try:
    app.include_router(preprocessing.router, prefix="/api/v1", tags=["Preprocessing"])
    app.include_router(template_parser.router, prefix="/api/v1", tags=["Template Parser"])
except ImportError:
    # These routers might not exist yet
    pass


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to add processing time headers and handle global error catching.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
    
    Returns:
        The response
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Measure request processing time
    start_time = time.time()
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Add processing time header
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
    except Exception as exc:
        # Log the exception
        logger.exception(f"Unhandled exception during request processing: {str(exc)}")
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Return standardized error response
        error_response = create_error_response(
            message="Internal server error",
            errors=[{"code": "INTERNAL_ERROR", "message": str(exc)}],
            request_id=request_id,
            processing_time_ms=process_time
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response,
            headers={"X-Process-Time-Ms": str(process_time), "X-Request-ID": request_id}
        )


@app.exception_handler(IDPBaseException)
async def handle_idp_exception(request: Request, exc: IDPBaseException):
    """
    Exception handler for IDP exceptions.
    
    Args:
        request: The incoming request
        exc: The exception
    
    Returns:
        Standardized error response
    """
    logger.warning(f"IDP exception: {str(exc)}")
    
    error_response = create_error_response(
        message=exc.message,
        errors=[{"code": exc.__class__.__name__, "message": exc.message}],
        request_id=getattr(request.state, "request_id", None)
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response
    )


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting up the IDP System API...")
    
    # Ensure cache directory exists
    os.makedirs("cache", exist_ok=True)
    
    # Register services in the DI container
    # This will be expanded as more services are implemented
    #container.register(IDocumentProcessor, DocumentProcessor, singleton=True)
    
    logger.info(f"IDP System API started in {'debug' if settings.DEBUG else 'production'} mode")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down the IDP System API...")
    
    # Perform cleanup tasks here
    
    logger.info("IDP System API shutdown complete")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": app.version}


# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0", 
#         port=8000, 
#         reload=settings.DEBUG, 
#         log_level=settings.LOG_LEVEL.lower()
#     )