# Multi-stage build for production
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.prod.txt requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.prod.txt

# Production stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/arbornote/.local/bin:$PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create non-root user first
RUN useradd --create-home --shell /bin/bash --uid 1000 arbornote

# Copy Python packages from builder
COPY --from=builder /root/.local /home/arbornote/.local

# Copy application code
COPY --chown=arbornote:arbornote . .

# Create necessary directories with proper permissions
RUN mkdir -p logs temp_files outputs checkpoints static/uploads \
    && chown -R arbornote:arbornote logs temp_files outputs checkpoints static \
    && chmod 755 logs temp_files outputs checkpoints static \
    && chmod +x *.sh 2>/dev/null || true

# Switch to non-root user
USER arbornote

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:4000/health || exit 1

# Start command with production optimizations
CMD ["gunicorn", \
     "--bind", "0.0.0.0:4000", \
     "--workers", "4", \
     "--worker-class", "gevent", \
     "--worker-connections", "1000", \
     "--timeout", "300", \
     "--keep-alive", "30", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--preload", \
     "--access-logfile", "logs/access.log", \
     "--error-logfile", "logs/error.log", \
     "--log-level", "info", \
     "app:app"]
