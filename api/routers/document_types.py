import cv2
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import Response
from core.document_type.document_classifier import DocumentVerification
from api.schemas.document_verification import DocumentVerificationResponse
from core.document_type.document_type_enums import DocumentType
from utils.image_utils import draw_bounding_box, image_to_base64, read_image
import base64

router = APIRouter()

verifier = DocumentVerification()
verifier.load_model()

@router.post("/verify-document/")
async def verify_document(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    # Read the image and preprocess it
    image_bytes = await file.read()
    image = read_image(image_bytes=image_bytes)
    result = verifier.verify_document(image=image)
    print(f"[INFO] Verification results: {result}")
    for res in result:
        if res.class_name == DocumentType.NO_CLASS:
            continue
        draw_bounding_box(image=image, bbox=res.bbox, text=res.class_name.value)
        
    # Convert from BGR to RGB before encoding
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_rgb = image_to_base64(image=image_rgb)
    
    # return DocumentVerificationResponse(**result)
    return DocumentVerificationResponse(
        image=image_rgb,
        verification_results=result
    ).to_response()