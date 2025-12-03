from PySide6.QtGui import QColor # Required for shadow color
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


# --- MODERN LIGHT STYLING CONSTANTS ---
PRIMARY_ACCENT_COLOR = "#007AFF"    # Professional Blue (e.g., Run)
PRIMARY_HOVER_COLOR = "#0056b3"
SECONDARY_ACCENT_COLOR = "#E74C3C"  # Red for critical (e.g., Reopen)
SECONDARY_HOVER_COLOR = "#C0392B"
BACKGROUND_COLOR = "#F4F6F8"      # Light Gray (almost white)
CONTAINER_BG_COLOR = "#FFFFFF"    # Pure White
TEXT_COLOR = "#000000"            # Pure Black (for text)
MUTED_TEXT_COLOR = "#7F8C8D"      # Gray for notes
BORDER_COLOR = "#DDE3E8"          # Light Gray border

# --- STYLING CONSTANTS ---
# (Derived from main_window.py)

# This block provides a high-specificity override for the DARK global stylesheet
# This requires widgets using this style to have setObjectName("toggleStyleButton")
DARK_TOGGLE_BUTTON_QSS = """
    QPushButton#toggleStyleButton {
        background-color: #FFD700; /* Yellow Gold bg */
        color: #000000; /* Black Text */
        border: 1px solid #FFD700;
        padding: 8px;
        font-weight: 500;
        border-radius: 5px;
    }
    QPushButton#toggleStyleButton:hover {
        background-color: #DAA520; /* Darker Gold hover */
    }
    QPushButton#toggleStyleButton:checked {
        background-color: #00bcd4; /* Cyan */
        color: white;
        border: 1px solid #00bcd4;
        font-weight: 600;
    }
    QPushButton#toggleStyleButton:checked:hover {
        background-color: #0097a7; /* Darker Cyan */
    }
"""

# This block provides a high-specificity override for the LIGHT global stylesheet
# This ensures the toggle buttons match the light theme
LIGHT_TOGGLE_BUTTON_QSS = f"""
    QPushButton#toggleStyleButton {{
        background-color: #ECF0F1; /* Light Gray bg */
        color: {TEXT_COLOR};
        border: 1px solid {BORDER_COLOR};
        padding: 8px;
        font-weight: 500;
        border-radius: 5px;
    }}
    QPushButton#toggleStyleButton:hover {{
        background-color: {BORDER_COLOR}; /* Darker Gray hover */
    }}
    QPushButton#toggleStyleButton:checked {{
        background-color: {PRIMARY_ACCENT_COLOR}; /* Blue */
        color: white;
        border: 1px solid {PRIMARY_ACCENT_COLOR};
        font-weight: 600;
    }}
    QPushButton#toggleStyleButton:checked:hover {{
        background-color: {PRIMARY_HOVER_COLOR}; /* Darker Blue */
    }}
"""


SUCCESS_BUTTON_STYLE = """
    QPushButton {
        background-color: #2ECC71; /* Green */
        color: white;
        font-weight: 600;
        border: none;
        padding: 8px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #27AE60; /* Darker Green */
    }
"""

SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #FF8C00; /* Orange */
        color: white;
        font-weight: 600;
        border: none;
        padding: 8px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #E67E00; /* Darker Orange */
    }
"""

SECTION_HEADER_STYLE = """
    font-weight: 600; 
    font-size: 15px; 
    margin-top: 8px; 
    margin-bottom: 2px;
    color: #9370DB; /* Purple */
"""

SUB_HEADER_STYLE = """
    font-weight: 600; 
    font-size: 13px; 
    color: #9370DB; /* Purple */
    margin-top: 5px;
"""

SCRIPT_HEADER_STYLE = """
    font-weight: 600; 
    font-size: 13px; 
    color: #9370DB; /* Purple */
"""

# Boolean Push Buttons
START_RED_STYLE = """
            QPushButton:checked {
                background-color: #06402B;
                color: white;
            }
            QPushButton {
                background-color: #8B0000;
                color: white;
            }
        """

START_GREEN_STYLE = """
    QPushButton:checked {
        background-color: #8B0000;
        color: white;
    }
    QPushButton {
        background-color: #06402B;
        color: white;
    }
"""

DARK_QSS = """
/* --- MODERN GLOBAL STYLE SHEET (QSS) - Corrected --- */
/* Accent Color: #00bcd4 (Cyan/Teal) */
/* Background Color: #1e1e1e (Soft Dark Gray) */
/* Secondary Background: #2d2d30 (Slightly Lighter) */

QWidget, QMainWindow, QDialog {
    background-color: #1e1e1e;
    color: #cccccc;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 10pt;
}

/* --- Buttons (Sleek, Lifted) --- */
QPushButton {
    /* Subtle gradient for a modern look */
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #00bcd4, stop: 1 #0097a7);
    color: white;
    border: none;
    padding: 10px 18px;
    border-radius: 6px;
    font-weight: 600;
}
QPushButton:hover {
    background: #00bcd4; /* Flat color on hover */
}
QPushButton:pressed {
    background: #00838f; /* Darker teal when pressed */
    padding-top: 12px; /* Simulate downward press */
    padding-bottom: 8px;
}
QPushButton:disabled {
    background-color: #3e3e3e;
    color: #888888;
}

/* --- Tab Widget Styling (Minimalist) --- */
QTabWidget::pane {
    border: 1px solid #3e3e3e;
    background-color: #1e1e1e;
    border-radius: 8px;
}
QTabBar::tab {
    background: #2d2d30;
    color: #aaaaaa;
    padding: 10px 20px;
    border: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1e1e1e;
    color: #00bcd4;
    border-bottom: 2px solid #00bcd4;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background: #3e3e3e;
    color: #cccccc;
}

/* --- Input Fields (Focus Highlighting) --- */
QLineEdit, QComboBox, QSpinBox, QTextEdit {
    background-color: #2d2d30;
    color: #cccccc;
    border: 1px solid #3e3e3e;
    padding: 8px;
    border-radius: 4px;
    selection-background-color: #00bcd4;
    selection-color: white;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
    border: 1px solid #00bcd4;
    background-color: #363639;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #3e3e3e;
    border-left-style: solid; 
}

/* --- Group Boxes (Cleaner Accent) --- */
QGroupBox {
    border: 1px solid #3e3e3e;
    margin-top: 25px;
    border-radius: 8px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 12px;
    background-color: #00bcd4;
    color: white;
    font-size: 11pt;
    border-radius: 4px;
}

/* --- Labels --- */
QLabel {
    color: #cccccc;
    background-color: transparent;
}

/* --- Scroll Area --- */
QScrollArea {
    border: 1px solid #3e3e3e;
    border-radius: 8px;
}

/* --- Scroll Bar Styling (Subtle Dark Mode) --- */
QScrollBar:vertical, QScrollBar:horizontal {
    border: none;
    background: #1e1e1e;
    width: 8px;
    height: 8px;
    margin: 0px; /* Reset margin */
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #555555;
    min-height: 20px;
    min-width: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #777777;
}
QScrollBar::add-line, QScrollBar::sub-line {
    border: none;
    background: none;
    height: 0px; /* Hide arrows */
    width: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

/* --- Header Widget Fix --- */
QWidget#header_widget {
    background-color: #2d2d30;
    border-bottom: 2px solid #00bcd4;
}

/* --- Custom Overrides --- */
#mainTitleLabel {
    font-size: 26px; 
    font-weight: 700; 
    padding-bottom: 5px;
    color: #9370DB; /* Purple */
}
#commandSelectLabel {
    font-weight: 600;
    color: #9370DB; /* Purple */
}

/* Override default button gradient */
#runButton {
    background: #00bcd4; /* Cyan */
    padding: 12px;
}
#runButton:hover {
    background: #0097a7; /* Darker Cyan */
}
#runButton:pressed {
    background: #00838f;
    padding-top: 13px; /* Match default press */
    padding-bottom: 11px;
}

#reopenButton {
    background: #9370DB; /* Purple */
}
#reopenButton:hover {
    background: #800080; /* Darker Purple */
}
#reopenButton:pressed {
    background: #4B0082;
    padding-top: 11px; /* Match default press */
    padding-bottom: 9px;
}


/* --- Theme Toggle Button --- */
QPushButton#themeToggleButton {
    background: transparent;
    border: none;
    color: #cccccc; /* Dark mode text color */
    font-size: 16px;
    padding: 0px;
}
QPushButton#themeToggleButton:hover {
    color: #00bcd4; /* Dark mode accent color */
}

/* --- Preview Text Box --- */
QTextEdit#previewTextEdit {
    background-color: #000000; /* Black background */
    color: #FFFFFF; /* White text */
    border: 1px solid #3e3e3e; /* Match other dark borders */
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 10px;
    border-radius: 5px;
}
"""

# --- ADD THE DARK TOGGLE BUTTON OVERRIDE TO DARK_QSS ---
DARK_QSS += DARK_TOGGLE_BUTTON_QSS


LIGHT_QSS = f"""
QWidget {{
    background-color: {BACKGROUND_COLOR};
    color: {TEXT_COLOR};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
}}
QLabel {{
    color: {TEXT_COLOR};
    padding-bottom: 2px;
}}
QComboBox, QTextEdit {{
    border: 1px solid {BORDER_COLOR};
    padding: 6px 8px;
    background-color: {CONTAINER_BG_COLOR};
    selection-background-color: {PRIMARY_ACCENT_COLOR};
    color: {TEXT_COLOR};
    border-radius: 5px; /* Rounded corners */
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: {BORDER_COLOR};
    border-left-style: solid; 
}}
QTabWidget::pane {{ /* The tab content area */
    border: 1px solid {BORDER_COLOR};
    background-color: {CONTAINER_BG_COLOR};
    border-radius: 5px;
    border-top-left-radius: 0; /* Align with tab */
}}
QTabBar::tab {{
    background: {BACKGROUND_COLOR};
    color: {MUTED_TEXT_COLOR};
    font-weight: 500;
    padding: 10px 18px;
    border: 1px solid {BORDER_COLOR};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{
    background: {CONTAINER_BG_COLOR};
    color: {PRIMARY_ACCENT_COLOR};
    font-weight: 600;
    /* Trick to make tab connect to pane */
    border-bottom: 2px solid {CONTAINER_BG_COLOR}; 
    margin-bottom: -2px;
}}
QTabBar::tab:hover {{
    color: {TEXT_COLOR};
}}
QScrollBar:vertical {{
    border: 1px solid {BORDER_COLOR};
    background: {CONTAINER_BG_COLOR}; /* White track background */
    width: 12px; 
    /* Margin reserves space for the arrow buttons */
    margin: 12px 0 12px 0; 
}}

QScrollBar::handle:vertical {{
    background: {TEXT_COLOR}; /* Dark Slate/Black for the handle */
    min-height: 20px;
    border-radius: 6px; 
}}

/* Style for the buttons holding the arrows (top and bottom) */
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    border: none;
    background: {TEXT_COLOR}; /* Black background for the arrow buttons */
    height: 12px;
    subcontrol-origin: margin;
}}

/* Position and shape the top button */
QScrollBar::sub-line:vertical {{ 
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    subcontrol-position: top; 
}}

/* Position and shape the bottom button */
QScrollBar::add-line:vertical {{
    border-bottom-left-radius: 5px;
    border-bottom-right-radius: 5px;
    subcontrol-position: bottom; 
}}

/* Explicitly define the Up Arrow (White arrow on Black button) */
QScrollBar::up-arrow:vertical {{
    image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDhMOCAxMkgxNkwxMiA4WiIgZmlsbD0id2hpdGUiLz4KPC9zdmc+);
    width: 8px; 
    height: 8px;
}}

/* Explicitly define the Down Arrow (White arrow on Black button) */
QScrollBar::down-arrow:vertical {{
    image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDE2TDE2IDEySDhMMTIgMTZaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2ZyA+);
    width: 8px; 
    height: 8px;
}}

/* The pages (area between handle and arrow buttons) should be transparent */
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* General Button Styling */
QPushButton {{
    color: white;
    font-weight: 600;
    border: none;
    padding: 10px 12px;
    border-radius: 5px;
}}
QPushButton:pressed {{
    /* Add a subtle press effect */
    padding-top: 11px;
    padding-bottom: 9px;
}}
QPushButton:disabled {{
    background-color: #BDC3C7; /* Muted gray */
    color: {TEXT_COLOR};
}}

/* --- Theme Toggle Button --- */
QPushButton#themeToggleButton {{
    background: transparent;
    border: none;
    color: {TEXT_COLOR};
    font-size: 16px;
    padding: 0px;
}}
QPushButton#themeToggleButton:hover {{
    color: {PRIMARY_ACCENT_COLOR};
}}

/* --- Custom Overrides --- */
#mainTitleLabel {{
    font-size: 26px; 
    font-weight: 700; 
    padding-bottom: 5px;
    color: {TEXT_COLOR}; 
}}
#commandSelectLabel {{
    font-weight: 600;
    color: {TEXT_COLOR};
}}

#runButton {{
    background-color: {PRIMARY_ACCENT_COLOR}; 
    padding: 12px;
}}
#runButton:hover {{ 
    background-color: {PRIMARY_HOVER_COLOR}; 
}}

#reopenButton {{ 
    background-color: {SECONDARY_ACCENT_COLOR}; 
}}
#reopenButton:hover {{ 
    background-color: {SECONDARY_HOVER_COLOR}; 
}}

/* --- Preview Text Box --- */
QTextEdit#previewTextEdit {{
    background-color: #FAFAFA;
    color: #333333;
    border: 1px solid {BORDER_COLOR};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 10px;
    border-radius: 5px;
}}

/* --- ADD THE LIGHT TOGGLE BUTTON OVERRIDE TO LIGHT_QSS --- */
{LIGHT_TOGGLE_BUTTON_QSS}
"""