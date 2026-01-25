"""
Path constants for the project.
"""
import os
from pathlib import Path

# Paths
path: Path = Path(os.getcwd())
parts: tuple[str, ...] = path.parts
try:
    root_dir = Path(*parts[: parts.index("WSmart-Route") + 1])
except ValueError:
    root_dir = Path(*parts[: parts.index("WSmartPlus-Route") + 1])
ROOT_DIR: Path = root_dir
ICON_FILE: str = os.path.join(ROOT_DIR, "assets", "images", "logo-wsmartroute-white.png")
