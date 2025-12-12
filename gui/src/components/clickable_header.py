from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


# Class to handle the custom clickable header functionality robustly
class ClickableHeaderWidget(QWidget):
    def __init__(self, toggle_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._toggle_function = toggle_function
        self.setCursor(Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        """Overrides the mouse press event to make the whole widget clickable."""
        # Ensure only left clicks are processed
        if event.button() == Qt.LeftButton:
            self._toggle_function()
        # Let the event propagate (if necessary, though the toggle function handles the UI change)
        super().mousePressEvent(event)