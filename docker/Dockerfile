# MLOps OCR Demo Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/custom models/registry logs

# Set development environment
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG

# Expose port
EXPOSE 7860

# Development command
CMD ["python", "gradio_app.py"]

# Stage 3: Production image
FROM base as production

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/custom models/registry logs

# Set production environment
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Production command
CMD ["python", "gradio_app.py"]

# Stage 4: Testing image
FROM development as testing

# Set testing environment
ENV ENVIRONMENT=testing
ENV LOG_LEVEL=DEBUG

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html"] 