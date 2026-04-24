"""
Path constants for the project.

This module provides platform-independent path resolution for the WSmart+ Route
project root and asset locations. Used by:
- GUI initialization (gui/src/windows/main_window.py) for icon loading
- All modules needing project-relative paths (configs, data, outputs)
- CLI entry points (main.py) for workspace detection

Root Directory Resolution
--------------------------
The ROOT_DIR is dynamically computed by searching upward from cwd until finding
the project root marker directory name. This supports:
- Running from any subdirectory (notebooks/, logic/test/, gui/)
- Multiple project clones (WSmart-Route, WSmartPlus-Route)
- Virtual environment isolation (paths work regardless of venv location)

Path Resolution Order:
1. Get current working directory
2. Search upward for "WSmart-Route" or "WSmartPlus-Route" in path parts
3. Set ROOT_DIR to that location
4. Derive asset paths relative to ROOT_DIR

Critical Files
--------------
- ICON_FILE: GUI application icon (used in window title bar, taskbar)

Attributes:
    path: Current working directory
    parts: Split path into components
    root_dir: Project root directory (absolute path)
    ROOT_DIR: Project root directory (absolute path)
    ICON_FILE: GUI application icon (used in window title bar, taskbar)

Example:
    >>> from logic.src.constants import ROOT_DIR, ICON_FILE
    >>> ROOT_DIR
    PosixPath('/home/user/Repositories/WSmart-Route')
    >>> ICON_FILE
    '/home/user/Repositories/WSmart-Route/assets/images/logo-wsmartroute-white.png'
"""

import os
import sys
from pathlib import Path

# Dynamic root directory resolution
# Searches upward from cwd for project root marker ("WSmart-Route" or "WSmartPlus-Route")
if getattr(sys, "frozen", False):
    path: Path = Path(sys._MEIPASS)  # Current working directory
else:
    path: Path = Path(__file__).parent.absolute()  # Current working directory


parts: tuple[str, ...] = path.parts  # Split path into components
try:
    root_dir = Path(*parts[: parts.index("logic")]).parent.absolute()
except ValueError:
    root_dir = Path(*parts[:-3]).absolute()

# Project root directory (absolute path)
# Example: /home/user/Repositories/WSmart-Route
ROOT_DIR: Path = root_dir

# Hydra configurations directory
CONFIGS_DIR: str = "configs"

# GUI application icon (PNG format, white logo on transparent background)
# Used in: PySide6 QMainWindow.setWindowIcon(), system tray, taskbar
# Dimensions: 512x512 px (scales down for UI)
ICON_FILE: str = os.path.join(ROOT_DIR, "assets", "images", "logo-wsmartroute-white.png")
