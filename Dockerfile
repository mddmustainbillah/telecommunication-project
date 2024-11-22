# Use Python 3.10 slim image as base
FROM python:3.10.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p mlruns data/raw data/interim data/processed models reports/metrics logs

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "telecommunication.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 