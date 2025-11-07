# --- MODERN LIGHT STYLING CONSTANTS ---
PRIMARY_ACCENT_COLOR = "#007AFF"    # Professional Blue (e.g., Run)
PRIMARY_HOVER_COLOR = "#0056b3"
SECONDARY_ACCENT_COLOR = "#E74C3C"  # Red for critical (e.g., Reopen)
SECONDARY_HOVER_COLOR = "#C0392B"
BACKGROUND_COLOR = "#F4F6F8"      # Light Gray (almost white)
CONTAINER_BG_COLOR = "#FFFFFF"    # Pure White
TEXT_COLOR = "#2C3E50"            # Dark Slate (for text)
MUTED_TEXT_COLOR = "#7F8C8D"      # Gray for notes
BORDER_COLOR = "#DDE3E8"          # Light Gray border

# --- STYLING CONSTANTS ---
# (Derived from main_window.py)
TOGGLE_BUTTON_STYLE = """
    QPushButton {
        background-color: #ECF0F1; /* Utility bg */
        color: #2C3E50; /* Text */
        border: 1px solid #DDE3E8; /* Border */
        padding: 8px;
        font-weight: 500;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #DDE3E8; /* Lighter hover */
    }
    QPushButton:checked {
        background-color: #007AFF; /* Primary Accent */
        color: white;
        border: 1px solid #007AFF;
        font-weight: 600;
    }
    QPushButton:checked:hover {
        background-color: #0056b3; /* Primary Hover */
    }
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
        background-color: #E74C3C; /* Red */
        color: white;
        font-weight: 600;
        border: none;
        padding: 8px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #C0392B; /* Darker Red */
    }
"""

SECTION_HEADER_STYLE = """
    font-weight: 600; 
    font-size: 15px; 
    margin-top: 8px; 
    margin-bottom: 2px;
    color: #2C3E50;
"""

SUB_HEADER_STYLE = """
    font-weight: 600; 
    font-size: 13px; 
    color: #2C3E50;
    margin-top: 5px;
"""

SCRIPT_HEADER_STYLE = """
    font-weight: 600; 
    font-size: 13px; 
    color: #2C3E50;
"""