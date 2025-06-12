# Chatterbox TTS Server - Optimized for Awesome-TTS Integration
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_HUB_CACHE=/app/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies (CPU-only)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create required directories for the application
RUN mkdir -p model_cache reference_audio outputs voices logs hf_cache ui static

# Pre-download the model to avoid startup delays (optional)
# Uncomment the following lines if you want to pre-download the model during build
# RUN python -c "
# import sys
# import signal
# import os
# def timeout_handler(signum, frame):
#     print('Model download timed out after 15 minutes')
#     sys.exit(1)
# signal.signal(signal.SIGALRM, timeout_handler)
# signal.alarm(900)  # 15 minutes
# try:
#     from chatterbox.tts import ChatterboxTTS
#     print('Pre-downloading Chatterbox model...')
#     model = ChatterboxTTS.from_pretrained(device='cpu')
#     print('Model pre-download completed successfully!')
#     signal.alarm(0)
# except Exception as e:
#     print(f'Model pre-download failed: {e}')
#     signal.alarm(0)
#     sys.exit(0)  # Continue anyway
# "

# Set proper permissions
RUN chmod -R 755 /app && \
    chmod -R 777 hf_cache model_cache reference_audio outputs voices logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the application will run on
EXPOSE 8000

# Command to run the application
CMD ["python", "server.py"]
