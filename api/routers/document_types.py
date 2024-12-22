from fastapi import APIRouter, UploadFile, HTTPException
from core.document_type.document_classifier import DocumentVerification
from api.schemas.document_verification import DocumentVerificationResponse
from config.config import get_settings

router = APIRouter()

# Load the precomputed base embeddings (replace with real embeddings)
settings = get_settings()

verifier = DocumentVerification(settings.MODEL_PATH , settings.BASE_EMBEDDINGS)

@router.post("/verify-document/")
async def verify_document(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    image_bytes = await file.read()
    result = verifier.verify(image_bytes)
    return DocumentVerificationResponse(class_name=result)
