"""
Light theme definitions.
"""

from gui.src.styles.colors import (
    BACKGROUND_COLOR,
    BORDER_COLOR,
    CONTAINER_BG_COLOR,
    MUTED_TEXT_COLOR,
    PRIMARY_ACCENT_COLOR,
    PRIMARY_HOVER_COLOR,
    SECONDARY_ACCENT_COLOR,
    SECONDARY_HOVER_COLOR,
    TEXT_COLOR,
)
from gui.src.styles.widgets import LIGHT_TOGGLE_BUTTON_QSS

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
    image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIg\
fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 8L8 12H16L12 8Z" fill="white"/>
</svg>);
    width: 8px;
    height: 8px;
}}

/* Explicitly define the Down Arrow (White arrow on Black button) */
QScrollBar::down-arrow:vertical {{
    image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIg\
fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 16L16 12H8L12 16Z" fill="white"/>
</svg> );
    width: 8px;
    height: 8px;
}}

/* The pages (area between handle and arrow buttons) should be transparent */
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* General Button Styling (BASE STYLE) */
QPushButton {{
    /* BASE STYLE derived from runButton aesthetic */
    background-color: {PRIMARY_ACCENT_COLOR}; /* Default Light Accent Color */
    color: white;
    font-weight: 600;
    border: none;
    padding: 10px 12px; /* Standard Padding */
    border-radius: 5px;
}}
QPushButton:hover {{
    background-color: {PRIMARY_HOVER_COLOR};
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
