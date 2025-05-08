# IDP-System

## Intelligent Document Processing

IDP-System is a powerful tool for automating document processing using advanced OCR models and AI techniques.

---

## Features

- **High-Accuracy OCR**: Extract text from images and PDFs with precision.
- **AI-Powered**: Leverage machine learning models for intelligent document classification and data extraction.
- **Customizable Pipelines**: Tailor the system to your specific document processing needs.
- **Scalable Architecture**: Designed to handle large-scale document processing tasks.
- **Multi-Language Support**: Process documents in various languages with ease.
- **Flexible Input Formats**: Supports a wide range of file types for processing.

---

## Supported Models

The system uses the following models for document processing:
- **Document Verification Model**: Ensures document authenticity and classification using deep learning.
- **OCR Models**: Includes Tesseract-based models, Google Vision API, AWS Textract, Azure Cognitive Services, and custom OCR implementations for high-accuracy text extraction.
- **AI Classification Models**: Utilizes advanced machine learning algorithms to categorize documents into predefined types.
- **Custom Models**: Easily integrate tailored models for specific document processing needs.

For a full list of available models and types, please refer to the enums in:
- `core/document_type/document_type_enums.py` for document-related models.
- `core/ocr_engine/ocr_engine_enums.py` for OCR engine types and languages.

---

## Supported OCR Engines

IDP-System supports multiple OCR engines to address diverse requirements:
- **Tesseract**: Open-source OCR engine for robust and flexible text recognition.
- **Surya**: Custom OCR engine leveraging specialized algorithms.
- **Paddle**: Deep learning based OCR engine for enhanced accuracy.
- **Easy OCR**: Simplified OCR solution for quick text extraction.
- **Google Vision API**: Cloud-based OCR for high accuracy and scalability.
- **AWS Textract**: Extracts text, tables, and forms from various documents.
- **Azure Cognitive Services**: Advanced OCR leveraging machine intelligence.
- **Custom OCR Engines**: Seamlessly integrate your own OCR solutions via the modular architecture.

---

## Supported Document Types

The system can handle a variety of document types:
- **ID Front** (NIDF)
- **ID Back** (NIDB)
- **Passport** (PASSPORT)
- **Card** (CARD)
- **Tax Card** (TAX)
- **Cheque** (CHEQUE)
- **Car Plate** (VEHICLE_PLATE)
- **Receipt** (RECEIPT)
- **Form** (FORM)
- **Table** (TABLE)
- **Text** (TEXT)
- **No Class** (NO_CLASS)

---

## Supported File Types

The system can process a wide array of file formats:
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, etc.
- **PDFs**: Supports both scanned and text-based `.pdf` files.
- **Office Documents**: `.docx`, `.xlsx`, `.pptx` (converted to images or PDFs as needed).
- **Other Formats**: Custom formats can be handled with appropriate preprocessing pipelines.

---

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/IDP-System.git
   cd IDP-System
   ```

2. Build the Docker image:
   ```bash
   docker build -t idp-system .
   ```

3. Run the Docker container:
   ```bash
   docker run -d -p 8080:8080 idp-system
   ```

---

## Usage

1. Access the API at `http://localhost:8080`.
2. Refer to the [API Documentation](docs/api_docs.md) for available endpoints and usage examples.
3. For detailed instructions, see the [User Guide](docs/user_guide.md).

---

## Documentation

- [API Documentation](docs/api_docs.md)
- [User Guide](docs/user_guide.md)
- [Architecture Overview](docs/architecture.md)

---

## Contributing

We welcome contributions! To get started:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork:
   ```bash
   git commit -m "Add feature-name"
   git push origin feature-name
   ```
4. Open a pull request.

Please ensure your code adheres to the project's coding standards and includes relevant tests.

---

## License

This project is licensed under the [CC0 1.0 Universal License](LICENSE).

---

## Support

For issues or questions, please open an issue in the [GitHub repository](https://github.com/your-repo/IDP-System/issues).

---

## Roadmap

- Add support for additional OCR languages.
- Integrate with cloud-based document storage solutions.
- Enhance AI models for better accuracy and speed.
- Expand support for additional file formats.

---

## Authors and Acknowledgments

- **Osama Mohamed** - AI Engineer
- Special thanks to contributors and the open-source community.

---

## Project Status

This project is actively maintained. Contributions and feedback are highly appreciated!

---

## Pipeline

Our customizable pipeline now includes:
- Document type detection and classification.
- Advanced OCR processing using updated engines.
- Data extraction, validation, and tailored post-processing.
