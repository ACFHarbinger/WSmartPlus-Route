"""
Widget styling constants (QSS snippets).
"""

from .colors import (
    BORDER_COLOR,
    PRIMARY_ACCENT_COLOR,
    PRIMARY_HOVER_COLOR,
    TEXT_COLOR,
)

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
