"""
GUI application-level constants.

Attributes:
    CTRL_C_TIMEOUT: Graceful shutdown timeout (seconds) on Ctrl+C
    APP_STYLES: Available Qt application styles
"""

from typing import List

# Time allowed for GUI to close cleanly before forced termination.
# Includes: saving state, closing database connections, stopping background workers.
CTRL_C_TIMEOUT: float = 2.0  # seconds

# Available Qt application styles (platform availability varies)
APP_STYLES: List[str] = ["fusion", "windows", "windowsxp", "macintosh"]
