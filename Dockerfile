# ncOS v22.0 - Zanlink Offline Engine
# Multi-stage build for optimized deployment

# Stage 1: Base dependencies
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Stage 2: Application
FROM base as app

# Copy application code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/data/cache \
    /app/data/zbar \
    /app/data/journals \
    /app/logs \
    /app/temp

# Set Python path
ENV PYTHONPATH=/app:/app/agents:$PYTHONPATH

# Environment variables
ENV NCOS_ENV=production
ENV NCOS_LOG_LEVEL=INFO
ENV NCOS_CACHE_DIR=/app/data/cache
ENV NCOS_DATA_DIR=/app/data

# Create non-root user
RUN useradd -m -u 1000 ncos && chown -R ncos:ncos /app
USER ncos

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001 8002 8003 8004

# Default command
CMD ["python", "ncos_launcher.py"]

# Stage 3: Offline data enrichment engine
FROM app as enrichment

# Additional dependencies for offline processing
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    ta==0.10.2 \
    vectorbt==0.25.5

# Copy enrichment scripts
COPY integrations/offline_enrichment.py /app/
COPY integrations/batch_processor.py /app/

# Enrichment-specific environment
ENV NCOS_MODE=enrichment
ENV BATCH_SIZE=1000
ENV ENRICHMENT_INTERVAL=300

# Run enrichment engine
CMD ["python", "offline_enrichment.py"]

# Stage 4: Development environment
FROM app as dev

# Install development dependencies
USER root
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    black==23.12.1 \
    flake8==6.1.0 \
    ipython==8.18.1

USER ncos

# Development command
CMD ["python", "-m", "ipython"]
