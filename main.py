# Main entry point
import uvicorn
from fastapi import FastAPI
from api.routers import input_source, ocr, document_types, roi, idp # Import your API routers
import logging 

app = FastAPI(title="OCR System API", description="API for Optical Character Recognition", version="0.1.0")

# Include API routers
app.include_router(input_source.router, prefix="/input_source", tags=["Input Source"])
app.include_router(document_types.router, prefix="/document_types", tags=["Document Types"])
app.include_router(roi.router, prefix="/roi", tags=["ROI"])
app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(idp.router, prefix="/idp", tags=["IDP"])

# Configure logging
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the OCR System API...")
    # config.load_config() # Load configuration on startup

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the OCR System API...")

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)

#python -m uvicorn main:app --host localhost --port 8000 --reload