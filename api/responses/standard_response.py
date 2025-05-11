"""
Standardized API response models for consistent API responses.
"""
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
import time

T = TypeVar('T')


class ErrorDetail(BaseModel):
    """
    Detailed error information with code, message, and path (if applicable).
    """
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    path: Optional[str] = Field(None, description="Path/field where the error occurred")


class Pagination(BaseModel):
    """
    Pagination information.
    """
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")


class Meta(BaseModel):
    """
    Response metadata.
    """
    timestamp: int = Field(default_factory=lambda: int(time.time()), description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class StandardResponse(GenericModel, Generic[T]):
    """
    Standardized API response format.
    
    Ensures consistent response format across all API endpoints.
    """
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field("", description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    errors: Optional[List[ErrorDetail]] = Field(None, description="List of errors, if any")
    meta: Meta = Field(default_factory=Meta, description="Response metadata")
    pagination: Optional[Pagination] = Field(None, description="Pagination information")


class SuccessResponse(StandardResponse[T]):
    """
    Standardized successful API response.
    """
    success: bool = True


class ErrorResponse(StandardResponse[None]):
    """
    Standardized error API response.
    """
    success: bool = False
    data: None = None


def create_success_response(
    data: Any = None,
    message: str = "Operation successful",
    request_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
    pagination: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Success message
        request_id: Unique request identifier
        processing_time_ms: Processing time in milliseconds
        pagination: Pagination information
    
    Returns:
        Standardized success response as a dictionary
    """
    meta = Meta(
        timestamp=int(time.time()),
        request_id=request_id,
        processing_time_ms=processing_time_ms
    )
    
    pagination_obj = None
    if pagination:
        pagination_obj = Pagination(
            page=pagination.get("page", 1),
            page_size=pagination.get("page_size", 10),
            total_items=pagination.get("total_items", 0),
            total_pages=pagination.get("total_pages", 0)
        )
    
    response = SuccessResponse(
        message=message,
        data=data,
        meta=meta,
        pagination=pagination_obj
    )
    
    return response.dict(exclude_none=True)


def create_error_response(
    message: str = "An error occurred",
    errors: Optional[List[Dict[str, str]]] = None,
    request_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        errors: List of detailed errors
        request_id: Unique request identifier
        processing_time_ms: Processing time in milliseconds
    
    Returns:
        Standardized error response as a dictionary
    """
    meta = Meta(
        timestamp=int(time.time()),
        request_id=request_id,
        processing_time_ms=processing_time_ms
    )
    
    error_details = None
    if errors:
        error_details = [
            ErrorDetail(
                code=error.get("code", "UNKNOWN_ERROR"),
                message=error.get("message", "Unknown error"),
                path=error.get("path")
            )
            for error in errors
        ]
    
    response = ErrorResponse(
        message=message,
        errors=error_details,
        meta=meta
    )
    
    return response.dict(exclude_none=True)
