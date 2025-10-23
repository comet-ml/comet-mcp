# Use Python 3.11 slim image for optimal size and compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY comet_mcp/ ./comet_mcp/

# Install the package and its dependencies
RUN pip install --no-cache-dir -e .

# Install additional dependencies for comet-ml compatibility
RUN pip install --no-cache-dir requests numpy

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash comet-mcp
USER comet-mcp

# Expose port for SSE transport (default: 8000)
EXPOSE 8000

# Health check for SSE transport
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Default command (SSE transport - recommended for Docker)
CMD ["comet-mcp", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands for different configurations:
# For custom host/port: CMD ["comet-mcp", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
