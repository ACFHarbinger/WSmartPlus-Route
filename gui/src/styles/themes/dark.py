"""
Dark theme definitions.
"""

from gui.src.styles.widgets import DARK_TOGGLE_BUTTON_QSS

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

/* --- General Buttons (Sleek, Lifted) --- */
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
