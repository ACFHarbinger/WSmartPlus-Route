# Use Python 3.9 as the base image
FROM python:3.9-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for Folium/Map rendering and UV
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libegl1 \
    libopengl0 \
    libxcb-cursor0 \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# We include gui and geo extras as the dashboard usually needs them
RUN uv sync --frozen --no-dev --extra gui --extra geo

# Copy the rest of your application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the application
ENTRYPOINT ["uv", "run", "streamlit", "run", "gui/src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
