"""
Custom header widgets for collapsible UI sections.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


class ClickableHeaderWidget(QWidget):
    """
    A widget that acts as a clickable header to toggle visibility of other components.
    """

    def __init__(self, toggle_function, *args, **kwargs):
        """
        Initialize the header with a toggle callback.

        Args:
            toggle_function (callable): Function to call when the header is clicked.
            *args: Variable length argument list for QWidget.
            **kwargs: Arbitrary keyword arguments for QWidget.
        """
        super().__init__(*args, **kwargs)
        self._toggle_function = toggle_function
        self.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]

    def mousePressEvent(self, event):
        """Overrides the mouse press event to make the whole widget clickable."""
        # Ensure only left clicks are processed
        if event.button() == Qt.LeftButton:  # type: ignore[attr-defined]
            self._toggle_function()
        # Let the event propagate (if necessary, though the toggle function handles the UI change)
        super().mousePressEvent(event)
