# Use the official Python slim image as the base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    TESSERACT_PATH="/usr/bin/tesseract" \
    TESSERACT_DIR="/usr/share/tesseract-ocr/5/tessdata"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy tessdata from assets/tessdata to Tesseract's tessdata directory
RUN mkdir -p ${TESSERACT_DIR} && \
    cp -r assets/tessdata/*.traineddata ${TESSERACT_DIR}/

# Create necessary directories
RUN mkdir -p cache models data/output logs

# Expose port
EXPOSE 8000

# Set default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]