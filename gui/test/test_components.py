
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from unittest.mock import MagicMock
from gui.src.components.clickable_header import ClickableHeaderWidget

def test_clickable_header_mouse_press(qapp):
    """Test that clicking the header triggers the toggle callback."""
    # Setup
    mock_toggle = MagicMock()
    header = ClickableHeaderWidget(mock_toggle)
    
    # Create Event (Left Click)
    # 0 is the timestamp, position is irrelevant for the logic as implemented
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        header.rect().center(),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier
    )
    
    # Execute
    header.mousePressEvent(event)
    
    # Assert
    mock_toggle.assert_called_once()

def test_clickable_header_right_click_ignored(qapp):
    """Test that right clicking does NOT trigger the toggle."""
    mock_toggle = MagicMock()
    header = ClickableHeaderWidget(mock_toggle)
    
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        header.rect().center(),
        Qt.RightButton, # Right Click
        Qt.RightButton,
        Qt.NoModifier
    )
    
    header.mousePressEvent(event)
    
    mock_toggle.assert_not_called()
