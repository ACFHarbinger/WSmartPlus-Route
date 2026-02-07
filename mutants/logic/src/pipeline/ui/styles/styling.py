# Copyright (c) WSmart-Route. All rights reserved.
"""
CSS styling and layout configuration for the dashboard.

Provides custom CSS and theme settings for consistent styling.
"""

# Custom CSS for the dashboard
CUSTOM_CSS = """
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    h1 {
        color: #1f77b4;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
    }

    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        min-width: 150px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 8px;
    }

    .kpi-card .label {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 8px;
    }

    .kpi-card .value {
        font-size: 24px;
        font-weight: bold;
    }

    /* Status indicators */
    .status-good {
        color: #28a745;
        font-weight: bold;
    }

    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }

    .status-error {
        color: #dc3545;
        font-weight: bold;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Map container */
    .map-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Metric container */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 20px;
    }

    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Data table styling */
    .dataframe {
        font-size: 14px;
    }

    .dataframe th {
        background-color: #f8f9fa;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1f77b4;
    }
</style>
"""


# Color palettes
CHART_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

ROUTE_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]

STATUS_COLORS = {
    "good": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "info": "#17a2b8",
}


def get_page_config() -> dict:
    """
    Get Streamlit page configuration.

    Returns:
        Dict with page config settings.
    """
    return {
        "page_title": "WSmart+ Control Tower",
        "page_icon": "🎛️",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }


def format_number(value: float, precision: int = 2) -> str:
    """Format a number with thousands separator and precision."""
    if abs(value) >= 1000:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1f}%"


def create_kpi_html(label: str, value: str, color: str = "#667eea") -> str:
    """
    Create HTML for a single KPI card.

    Args:
        label: Metric label.
        value: Formatted value string.
        color: Background gradient start color.

    Returns:
        HTML string.
    """
    return (
        f'<div style="background: linear-gradient(135deg, {color} 0%, #764ba2 100%); '
        f"border-radius: 12px; padding: 20px; min-width: 140px; text-align: center; "
        f'color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
        f'<div style="font-size: 13px; opacity: 0.9; margin-bottom: 6px;">{label}</div>'
        f'<div style="font-size: 22px; font-weight: bold;">{value}</div>'
        f"</div>"
    )


def create_kpi_row(metrics: dict) -> str:
    """
    Create HTML for a row of KPI cards.

    Args:
        metrics: Dict of label -> value.

    Returns:
        HTML string for the row.
    """
    cards = []
    colors = ["#667eea", "#28a745", "#17a2b8", "#ffc107", "#dc3545", "#6f42c1"]

    for i, (label, value) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        if isinstance(value, float):
            formatted = format_number(value)
        else:
            formatted = str(value)
        cards.append(create_kpi_html(label, formatted, color))

    return f'<div style="display: flex; flex-wrap: wrap; gap: 12px;">{"".join(cards)}</div>'
