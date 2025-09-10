# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs temp_files outputs checkpoints

# Set file permissions
RUN chmod +x start_production.sh || true
RUN chmod 755 logs temp_files outputs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash arbornote
RUN chown -R arbornote:arbornote /app
USER arbornote

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:4000/health || exit 1

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:4000", "--workers", "4", "--worker-class", "gevent", "--timeout", "300", "--access-logfile", "logs/access.log", "--error-logfile", "logs/error.log", "app:app"]
