# Use Python 3.9 as the base image (matching project requirement)
FROM python:3.9-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for GUI (Qt/PySide6), GIS (Folium/OSMnx), and testing (xvfb)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libegl1 \
    libopengl0 \
    libxcb-cursor0 \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    xvfb \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-shape0 \
    libasound2 \
    libpulse0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user for development
RUN useradd -m -s /bin/bash vscode
USER vscode
WORKDIR /app

# The devcontainer.json postCreateCommand will handle 'uv sync'
# but we can pre-set the path
ENV PATH="/app/.venv/bin:$PATH"

# Default shell
CMD ["/bin/bash"]
