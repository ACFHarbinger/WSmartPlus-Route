"""
Visual effects utilities.
"""

from PySide6.QtGui import QColor  # Required for shadow color
from PySide6.QtWidgets import QGraphicsDropShadowEffect


def apply_shadow_effect(widget, color_hex="#000000", radius=10, x_offset=0, y_offset=4):
    """Creates and applies a QGraphicsDropShadowEffect to a given widget."""
    shadow = QGraphicsDropShadowEffect(widget)

    # 1. Set the color (black with transparency)
    shadow.setColor(QColor(color_hex))

    # 2. Set the blur radius (controls the softness/spread)
    shadow.setBlurRadius(radius)

    # 3. Set the offset (controls the shadow position, similar to CSS x/y)
    shadow.setOffset(x_offset, y_offset)

    # 4. Apply the effect to the widget
    widget.setGraphicsEffect(shadow)
    return shadow
