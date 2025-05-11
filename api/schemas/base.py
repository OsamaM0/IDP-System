from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')

class ResponseStatus:
    SUCCESS = "success"
    ERROR = "error"

class BaseResponse(BaseModel, Generic[T]):
    """Base response model for standardized API responses"""
    status: str = Field(..., description="Response status (success or error)")
    message: str = Field(..., description="Human-readable message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    data: Optional[T] = Field(None, description="Response data")
    
class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str = Field(..., description="Error code")
    detail: str = Field(..., description="Detailed error message")
    source: Optional[str] = Field(None, description="Error source or location")

class ErrorResponse(BaseResponse[List[ErrorDetail]]):
    """Standardized error response"""
    status: str = ResponseStatus.ERROR
    errors: List[ErrorDetail] = Field(default_factory=list, description="List of errors")

class SuccessResponse(BaseResponse[T]):
    """Standardized success response"""
    status: str = ResponseStatus.SUCCESS
    
class PaginatedResponse(SuccessResponse[List[T]]):
    """Paginated response for lists"""
    total: int = Field(..., description="Total number of items")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
