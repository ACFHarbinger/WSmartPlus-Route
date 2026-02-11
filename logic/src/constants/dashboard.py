"""
Dashboard visualization color schemes.

This module defines color palettes for the simulation results dashboard.
Used by:
- gui/src/windows/ts_results_window.py (simulation dashboard)
- logic/src/utils/logging/plotting/ (matplotlib chart generation)
- Folium map visualizations in notebooks

Color Selection Rationale
--------------------------
Colors are chosen from the ColorBrewer2 qualitative palette for:
- Maximum visual distinction between routes (8+ routes distinguishable)
- Colorblind-safe palette (tested for deuteranopia, protanopia)
- Print-friendly (maintains distinction in grayscale)

Usage Example
-------------
    >>> from logic.src.constants.dashboard import ROUTE_COLORS
    >>> for route_id, tour in enumerate(tours):
    >>>     color = ROUTE_COLORS[route_id % len(ROUTE_COLORS)]
    >>>     plt.plot(tour, color=color, label=f"Vehicle {route_id}")
"""

# Color palette for different vehicles/routes
# ColorBrewer2 "Set1" qualitative palette (8 colors)
# Optimized for: categorical data, print, colorblind-safe
# Cycling: Use modulo operator when >8 routes: color = ROUTE_COLORS[route_id % 8]
ROUTE_COLORS = [
    "#e41a1c",  # Red - Vehicle 0, high visibility
    "#377eb8",  # Blue - Vehicle 1, primary color
    "#4daf4a",  # Green - Vehicle 2, success color
    "#984ea3",  # Purple - Vehicle 3, distinct from RGB
    "#ff7f00",  # Orange - Vehicle 4, warm color
    "#ffff33",  # Yellow - Vehicle 5, high visibility (avoid white backgrounds)
    "#a65628",  # Brown - Vehicle 6, earth tone
    "#f781bf",  # Pink - Vehicle 7, complements purple
]

# Bin status colors for dashboard and map markers
# Bootstrap color scheme for semantic consistency with GUI
# Used in: Folium map popups, bin state heatmaps, status legends
BIN_COLORS = {
    "served": "#28a745",  # Green (Bootstrap success) - bin was collected today
    "pending": "#dc3545",  # Red (Bootstrap danger) - bin needs collection
    "must_go": "#fd7e14",  # Orange (Bootstrap warning) - bin was in must_go selection
    "depot": "#007bff",  # Blue (Bootstrap primary) - depot/warehouse location
}
