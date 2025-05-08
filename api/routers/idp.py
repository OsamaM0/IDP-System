import json
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
from typing import Optional
from api.controllers.idp_controller import IDPController
from core.document_type.document_type_enums import DocumentType
from core.ocr_engine.ocr_engine_enums import OCREngineType, OCRLanguage
from core.input.file_input import FileInput
from core.input.url_input import UrlInput
from core.input.scanner_input import ScannerInput
from core.factories.input_source_factory import InputSourceFactory
from utils.image_utils import read_image

router = APIRouter()


class InputDataSchema(BaseModel):
    @router.post("/process-idp")
    async def process_idp_image(
        input_data: Optional[str] = None,
        input_type: Optional[str] = "auto",
        ocr_engine_type: Optional[OCREngineType] = OCREngineType.PADDLE,
        language: Optional[OCRLanguage] = OCRLanguage.ARABIC,  
        doc_type: Optional[DocumentType] = None,
        file: Optional[UploadFile] = File(None),
        
    ):
        """
        API endpoint to process input data using IDPController and return JSON data.
        
        Args:
            input (InputDataSchema, optional): Input data, type, OCR engine, and language.
            file (UploadFile, optional): Uploaded image file (for raw bytes).
        
        Returns:
            dict: JSON-compatible dictionary containing extracted data.
        """
        try:
            # Determine input source
            if file:
                # Handle uploaded file (raw bytes)
                input_type = "bytes"
                image_data = await file.read()

            elif  input_data:
                # Handle input from schema (file path, URL, scanner)
                input_type = input_type

                if input_type == "auto":
                    # Let InputSourceFactory decide based on input_data
                    input_source = InputSourceFactory.create_input_source(input_data)
                    image_data = input_source.load_image()
                elif input_type == "file":
                    input_source = FileInput(input_data)
                    image_data = input_source.load_image()
                elif input_type == "url":
                    input_source = UrlInput(input_data)
                    image_data = input_source.load_image()
                elif input_type == "scanner":
                    input_source = ScannerInput()
                    image_data = await input_source.load_image()
                else:
                    raise ValueError(f"Unsupported input_type: {input_type}")
            else:
                raise ValueError("No input provided. Provide either input_data or a file.")

            # Convert image data to numpy array
            image = read_image(image_bytes=image_data)
            image_np = np.array(image)

            # Apply IDP processing
            result = IDPController.apply_idp(
                image=image_np,
                ocr_engine_type=ocr_engine_type,
                language=language,
                doc_type=doc_type
            )
            print(result)
            # Convert result to JSON-compatible format
            json_data = {
                "status": "success",
                "input_type": input_type,
                "ocr_engine_type": ocr_engine_type.value,
                "language": language.value,
                "extracted_data": result
            }

            # Save to a JSON file
            with open("idp_output.json", "w") as f:
                json.dump(json_data, f, indent=4)

            return json_data

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")