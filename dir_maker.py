import os
import shutil

def create_project_structure(project_name="ocr_system"):
    """Creates the complete project directory structure."""

    base_dir = project_name

    dirs = [
        os.path.join(base_dir, "api", "controllers"),
        os.path.join(base_dir, "api", "routers"),
        os.path.join(base_dir, "api", "models"),
        os.path.join(base_dir, "api", "schemas"),
        os.path.join(base_dir, "core", "input"),
        os.path.join(base_dir, "core", "document_type"),
        os.path.join(base_dir, "core", "preprocessing"),
        os.path.join(base_dir, "core", "ocr_engine"),
        os.path.join(base_dir, "core", "template_parser"),
        os.path.join(base_dir, "core", "postprocessing"),
        os.path.join(base_dir, "core", "output"),
        os.path.join(base_dir, "core", "models"),
        os.path.join(base_dir, "core", "factories"),
        os.path.join(base_dir, "config", "templates"),
        os.path.join(base_dir, "data", "sample_images"),
        os.path.join(base_dir, "data", "training_data"),
        os.path.join(base_dir, "docs"),
        os.path.join(base_dir, "examples"),
        os.path.join(base_dir, "ml", "document_classifier"),
        os.path.join(base_dir, "ml", "template_matcher"),
        os.path.join(base_dir, "ml", "training_scripts"),
        os.path.join(base_dir, "scripts"),
        os.path.join(base_dir, "tests", "core"),
        os.path.join(base_dir, "tests", "api"),
        os.path.join(base_dir, "utils"),
        os.path.join(base_dir, "db"),
    ]

    files = {
        os.path.join(base_dir, "api", "__init__.py"): "",
        os.path.join(base_dir, "api", "controllers", "__init__.py"): "",
        os.path.join(base_dir, "api", "routers", "__init__.py"): "",
        os.path.join(base_dir, "api", "models", "__init__.py"): "",
        os.path.join(base_dir, "api", "schemas", "__init__.py"): "",
        os.path.join(base_dir, "core", "__init__.py"): "",
        os.path.join(base_dir, "core", "input", "__init__.py"): "",
        os.path.join(base_dir, "core", "document_type", "__init__.py"): "",
        os.path.join(base_dir, "core", "preprocessing", "__init__.py"): "",
        os.path.join(base_dir, "core", "ocr_engine", "__init__.py"): "",
        os.path.join(base_dir, "core", "template_parser", "__init__.py"): "",
        os.path.join(base_dir, "core", "postprocessing", "__init__.py"): "",
        os.path.join(base_dir, "core", "output", "__init__.py"): "",
        os.path.join(base_dir, "core", "models", "__init__.py"): "",
        os.path.join(base_dir, "core", "factories", "__init__.py"): "",
        os.path.join(base_dir, "config", "__init__.py"): "",
        os.path.join(base_dir, "ml", "__init__.py"): "",
        os.path.join(base_dir, "tests", "__init__.py"): "",
        os.path.join(base_dir, "utils", "__init__.py"): "",
        os.path.join(base_dir, "db", "__init__.py"): "",
        os.path.join(base_dir, "config", "config.py"): "# Configuration settings",
        os.path.join(base_dir, "config", "settings.ini"): "[DEFAULT]",
        os.path.join(base_dir, "docs", "api_docs.md"): "# API Documentation",
        os.path.join(base_dir, "docs", "architecture.md"): "# Architecture Overview",
        os.path.join(base_dir, "docs", "user_guide.md"): "# User Guide",
        os.path.join(base_dir, "examples", "ocr_example.py"): "# Example usage",
        os.path.join(base_dir, "scripts", "preprocess_data.py"): "# Data preprocessing script",
        os.path.join(base_dir, ".dockerignore"): "__pycache__\n.DS_Store\n.env\n.venv/\n*.pyc\n*.egg-info/\ndist/\nbuild/\ndata/training_data/*\ndocs/_build/\n*.log\n.pytest_cache/\n.coverage",
        os.path.join(base_dir, ".gitignore"): "__pycache__\n.DS_Store\n.env\n.venv/\n*.pyc\n*.egg-info/\ndist/\nbuild/\ndata/training_data/*\ndocs/_build/\n*.log\n.pytest_cache/\n.coverage",
        os.path.join(base_dir, "Dockerfile"): "# Dockerfile content",
        os.path.join(base_dir, "LICENSE"): "MIT License",
        os.path.join(base_dir, "README.md"): "# OCR System\n",
        os.path.join(base_dir, "requirements.txt"): "fastapi\nuvicorn[standard]\ngunicorn\npython-multipart\nPillow\nopencv-python\npytesseract\nrequests\nnumpy\nscikit-learn",
        os.path.join(base_dir, "setup.py"): "from setuptools import setup, find_packages\n\nsetup(\n    name='ocr_system',\n    version='0.1.0',\n    packages=find_packages(),\n    install_requires=[],\n    python_requires='>=3.10',\n)",
        os.path.join(base_dir, "main.py"): "# Main entry point",
    }

    try:
        os.makedirs(base_dir, exist_ok=True)  # Create the base directory

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

        for file_path, file_content in files.items():
            with open(file_path, "w") as f:
                f.write(file_content)

        print(f"Project structure created in '{base_dir}'")

    except OSError as e:
        print(f"Error creating project structure: {e}")

if __name__ == "__main__":
    create_project_structure()