# ASR Demo for Disordered Speech - Docker Image
# 
# This Dockerfile builds a containerized version of the ASR demo application
# with the ASR model pre-downloaded for offline use.
#
# Build: docker build -t asr-disordered-speech .
# Run:   docker run -p 8501:8501 asr-disordered-speech

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_python.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_python.txt

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY pytest.ini .

# Create models directory
RUN mkdir -p /app/models

# Pre-download the ASR model during build
# This ensures the model is available offline
RUN python -c "\
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC; \
print('Downloading wav2vec2-base-960h model...'); \
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h', cache_dir='/app/models'); \
Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h', cache_dir='/app/models'); \
print('Model downloaded successfully!')"

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
