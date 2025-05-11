# User Guide

## Getting Started

This guide will help you get started with the IDP-System for intelligent document processing.

### Installation

#### Docker Installation (Recommended)

1. Pull the Docker image:
   ```bash
   docker pull your-registry/idp-system:latest
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 your-registry/idp-system:latest
   ```

3. Verify the installation by accessing the API documentation at:
   ```
   http://localhost:8000/docs
   ```

#### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/IDP-System.git
   cd IDP-System
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start the application:
   ```bash
   python main.py
   ```

### Basic Usage

#### Processing a Document

To process a document through the API:

1. Use the `/idp/process-idp` endpoint with a document file:

```bash
curl -X POST "http://localhost:8000/idp/process-idp" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/document.jpg" \
  -F "ocr_engine_type=PADDLE" \
  -F "language=ARABIC"
```

2. Or provide a URL to the document:

```bash
curl -X POST "http://localhost:8000/idp/process-idp" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": "https://example.com/document.jpg",
    "input_type": "url",
    "ocr_engine_type": "PADDLE",
    "language": "ARABIC"
  }'
```

## API Reference

### Main Endpoints

#### Document Processing

- **POST** `/idp/process-idp`
  - Process a document and extract structured information
  - Parameters:
    - `file`: The document image file (optional)
    - `input_data`: URL or path to document (optional)
    - `input_type`: Type of input ('file', 'url', 'scanner', 'auto')
    - `ocr_engine_type`: OCR engine to use (default: PADDLE)
    - `language`: Language for OCR processing (default: ARABIC)
    - `doc_type`: Document type (optional, auto-detected if not provided)

#### Document Classification

- **POST** `/document_types/verify`
  - Identify the type of document
  - Parameters:
    - `file`: The document image file

#### Region of Interest Extraction

- **POST** `/roi/roi`
  - Extract regions of interest from a document
  - Parameters:
    - `file`: The document image file
    - `document_type`: Type of document

#### OCR Processing

- **POST** `/ocr/ocr`
  - Perform OCR on an image
  - Parameters:
    - `file`: The image file
    - `ocr_engine_type`: OCR engine to use
    - `language`: Language for OCR processing

## Supported Document Types

The system supports the following document types:
- **NIDF**: National ID Front
- **NIDB**: National ID Back
- **PASSPORT**: Passport
- **TAX**: Tax Card
- **VEHICLE_PLATE**: Vehicle License Plate
- **RECEIPT**: Receipt
- **FORM**: Form
- **TABLE**: Table
- **TEXT**: Generic Text Document
- **NO_CLASS**: Unclassified Document

## Troubleshooting

### Common Issues

1. **Poor OCR Results**:
   - Ensure the image is clear and well-lit
   - Try a different OCR engine
   - Use document-specific preprocessing

2. **Document Type Not Detected**:
   - Specify the document type manually
   - Ensure the document is properly oriented
   - Check that the document is fully visible in the image

3. **API Errors**:
   - Check the error message in the response
   - Verify your API key is valid
   - Check server logs for detailed error information

### Getting Help

If you encounter any issues not covered in this guide, please:

1. Check the detailed API documentation at `/docs`
2. Review the logs for error messages
3. Open an issue in the project repository
4. Contact the support team

## Advanced Configuration

For advanced configuration options, see the [Configuration Guide](configuration.md).