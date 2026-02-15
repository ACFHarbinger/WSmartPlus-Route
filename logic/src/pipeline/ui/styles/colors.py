# Copyright (c) WSmart-Route. All rights reserved.
"""
Color palettes and page configuration for the dashboard.

Provides semantic color mappings for KPI cards, charts, routes, and status indicators,
plus the Streamlit page configuration.
"""

from typing import Dict, List, Tuple

# Semantic color palette for KPI cards — each metric gets a distinct gradient
KPI_COLORS: Dict[str, Tuple[str, str]] = {
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
    # Cumulative summary metrics
    "Total Profit": ("#43a047", "#2e7d32"),
    "Total Distance (km)": ("#1e88e5", "#1565c0"),
    "Total Waste (kg)": ("#8e24aa", "#6a1b9a"),
    "Total Overflows": ("#e53935", "#c62828"),
    "Total Cost": ("#fb8c00", "#ef6c00"),
    "Avg Efficiency": ("#039be5", "#0277bd"),
}

# Fallback gradient cycle for unknown metric names
KPI_FALLBACK_COLORS: List[Tuple[str, str]] = [
    ("#667eea", "#5a67d8"),
    ("#43a047", "#2e7d32"),
    ("#039be5", "#0277bd"),
    ("#fb8c00", "#ef6c00"),
    ("#e53935", "#c62828"),
    ("#7e57c2", "#5e35b1"),
]

# Color palettes
CHART_COLORS: List[str] = [
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

ROUTE_COLORS: List[str] = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]

STATUS_COLORS: Dict[str, str] = {
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
        "page_icon": "\U0001f39b\ufe0f",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }
