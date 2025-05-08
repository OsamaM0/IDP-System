from pydantic import BaseModel, Field, validator
from typing import Literal, Union

# Preprocessor Input Schema using Pydantic
class PreprocessorRequestSchema(BaseModel):
    file_type: Literal['nid', 'tax', 'passport'] = Field(
        ..., 
        description="The type of file to be preprocessed. Allowed values are 'nid', 'tax', 'passport'.",
        example="nid"
    )
    image: Union[bytes, str] = Field(
        ..., 
        description="The image to be preprocessed. Accepts binary (bytes) data or base64-encoded string.",
        example="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCA..."
    )

    @validator("image")
    def validate_image(cls, value):
        if isinstance(value, str) and not value.startswith("data:image/"):
            raise ValueError("If provided as a string, the image must be base64-encoded with a 'data:image/' prefix.")
        return value


# Endpoint Response Schema
class PreprocessorResponseSchema(BaseModel):
    image: str = Field(
        ..., 
        description="The preprocessed image as a base64-encoded string.",
        example="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCA..."
    )
