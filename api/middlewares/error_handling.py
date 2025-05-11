import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging

from api.schemas.base import ErrorResponse, ErrorDetail

logger = logging.getLogger(__name__)

async def error_handler_middleware(request: Request, call_next):
    """Middleware to catch and format all unhandled exceptions"""
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"Unhandled exception: {str(exc)}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                message="An unexpected error occurred",
                errors=[
                    ErrorDetail(
                        code="internal_server_error",
                        detail=str(exc),
                        source="server"
                    )
                ]
            ).dict()
        )

async def handle_validation_exception(request: Request, exc: RequestValidationError):
    """Handler for FastAPI request validation errors"""
    errors = []
    
    for error in exc.errors():
        errors.append(
            ErrorDetail(
                code="validation_error",
                detail=error["msg"],
                source=".".join(str(loc) for loc in error["loc"]) if error.get("loc") else None
            )
        )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            message="Validation error",
            errors=errors
        ).dict()
    )
