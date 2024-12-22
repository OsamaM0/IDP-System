from pydantic import BaseModel, Field

# Input Schema using Pydantic
class InputDataSchema(BaseModel):
    input_data: str = Field(..., description="Input data, e.g., a file path, URL, or 'scanner' identifier")

# Endpoint Response Schema
class ResponseSchema(BaseModel):
    input_type: str
    message: str