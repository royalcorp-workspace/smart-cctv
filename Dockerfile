# ==============================================================================
# Smart CCTV 2.0 - Production Container Image
# Multi-Camera AI Monitoring, Dwell Tracking & FastAPI Web Dashboard
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr for real-time logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV, OpenVINO, FFmpeg, and audio workers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies without caching wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . /app

# Ensure directories for storage, data, and logs exist
RUN mkdir -p /app/storage /app/data /app/logs /app/cameras

# Expose FastAPI Web Dashboard port
EXPOSE 8000

# Default entrypoint for 24/7 headless production execution
CMD ["python", "main.py", "--headless"]
