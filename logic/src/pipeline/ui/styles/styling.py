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
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    h1 {
        color: #1a73e8;
        border-bottom: 3px solid #e8eaed;
        padding-bottom: 12px;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #202124;
        font-weight: 600;
        margin-top: 1.5rem;
    }

    h3 {
        color: #3c4043;
        font-weight: 600;
    }

    /* KPI Card styling */
    .kpi-card {
        border-radius: 14px;
        padding: 18px 20px;
        min-width: 140px;
        text-align: center;
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
    }

    .kpi-card .label {
        font-size: 13px;
        opacity: 0.92;
        margin-bottom: 6px;
        font-weight: 500;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    .kpi-card .value {
        font-size: 24px;
        font-weight: 700;
    }

    /* Status indicators */
    .status-good {
        color: #0d904f;
        font-weight: bold;
    }

    .status-warning {
        color: #e37400;
        font-weight: bold;
    }

    .status-error {
        color: #c5221f;
        font-weight: bold;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Map container */
    .map-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Metric container */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 20px;
    }

    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #e8eaed;
    }

    /* Data table styling */
    .dataframe {
        font-size: 14px;
    }

    .dataframe th {
        background-color: #f1f3f4;
        font-weight: 600;
    }

    /* Section divider */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #dadce0, transparent);
        margin: 1.5rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a73e8;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 500;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        h1 {
            color: #8ab4f8;
            border-bottom-color: #3c4043;
        }

        h2, h3 {
            color: #e8eaed;
        }

        .chart-container {
            background: #292a2d;
            border-color: #3c4043;
        }

        .dataframe th {
            background-color: #3c4043;
        }

        section[data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }
    }
</style>
"""


# Semantic color palette for KPI cards — each metric gets a distinct gradient
KPI_COLORS = {
    "Day": ("#5c6bc0", "#3949ab"),
    "Profit": ("#43a047", "#2e7d32"),
    "Distance (km)": ("#1e88e5", "#1565c0"),
    "Waste (kg)": ("#8e24aa", "#6a1b9a"),
    "Overflows": ("#e53935", "#c62828"),
    "Collections": ("#00897b", "#00695c"),
    "Waste Lost (kg)": ("#f4511e", "#d84315"),
    "Efficiency (kg/km)": ("#039be5", "#0277bd"),
    "Cost": ("#fb8c00", "#ef6c00"),
    "Epochs": ("#5c6bc0", "#3949ab"),
    "Steps": ("#7e57c2", "#5e35b1"),
    "Latest Loss": ("#e53935", "#c62828"),
    "Best Loss": ("#43a047", "#2e7d32"),
    "Latest Val": ("#fb8c00", "#ef6c00"),
    "Best Val": ("#00897b", "#00695c"),
    "Time/Epoch (s)": ("#546e7a", "#37474f"),
    "Total Runs": ("#5c6bc0", "#3949ab"),
    "Best Latency (s)": ("#43a047", "#2e7d32"),
    "Best Throughput": ("#039be5", "#0277bd"),
    "Benchmark Types": ("#7e57c2", "#5e35b1"),
}

# Fallback gradient cycle for unknown metric names
KPI_FALLBACK_COLORS = [
    ("#667eea", "#5a67d8"),
    ("#43a047", "#2e7d32"),
    ("#039be5", "#0277bd"),
    ("#fb8c00", "#ef6c00"),
    ("#e53935", "#c62828"),
    ("#7e57c2", "#5e35b1"),
]

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


def create_kpi_html(label: str, value: str, color: str = "#667eea", color_end: str = "#5a67d8") -> str:
    """
    Create HTML for a single KPI card.

    Args:
        label: Metric label.
        value: Formatted value string.
        color: Background gradient start color.
        color_end: Background gradient end color.

    Returns:
        HTML string.
    """
    return (
        f'<div class="kpi-card" style="background: linear-gradient(135deg, {color} 0%, {color_end} 100%);">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"</div>"
    )


def create_kpi_row(metrics: dict) -> str:
    """
    Create HTML for a row of KPI cards with semantic colors.

    Args:
        metrics: Dict of label -> value.

    Returns:
        HTML string for the row.
    """
    cards = []

    for i, (label, value) in enumerate(metrics.items()):
        if label in KPI_COLORS:
            color, color_end = KPI_COLORS[label]
        else:
            fallback = KPI_FALLBACK_COLORS[i % len(KPI_FALLBACK_COLORS)]
            color, color_end = fallback

        formatted = format_number(value) if isinstance(value, float) else str(value)
        cards.append(create_kpi_html(label, formatted, color, color_end))

    return f'<div style="display: flex; flex-wrap: wrap; gap: 12px;">{"".join(cards)}</div>'
