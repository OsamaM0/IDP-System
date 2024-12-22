# ocr_response.py
from pydantic import BaseModel

class DocumentVerificationResponse(BaseModel):
    class_name: str
