# SciML Framework Production Deployment
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY sciml/ ./sciml/
COPY tests/ ./tests/

# Create non-root user for security
RUN useradd -m -u 1000 opifex && chown -R opifex:opifex /app
USER opifex

# Expose default serving port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV JAX_PLATFORM_NAME=cpu
ENV JAX_ENABLE_X64=true

# Default command
CMD ["python", "-m", "sciml.deployment.server"]
