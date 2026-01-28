# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies required for Folium/Map rendering and UV
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev

# Copy the rest of your application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the application
ENTRYPOINT ["uv", "run", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
