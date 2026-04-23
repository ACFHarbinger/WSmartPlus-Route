"""Metric and route visualization package.

This package provides high-level plotting utilities for monitoring training
progress and visualizing routing solutions. It includes tools for 2D line
charts, interactive heatmaps, neural attention maps, and geographic/cartesian
route rendering.

Attributes:
    plot_linechart: Main entry point for static metric charts.
    draw_graph: Coordinate-free graph visualization.
    plot_tsp: Renders single-vehicle tours.
    plot_vehicle_routes: Renders multi-vehicle VRP solutions.
    discrete_cmap: Utility for vehicle-indexed coloring.
    plot_attention_maps_wrapper: Heatmap generator for transformer attention.
    visualize_interactive_plot: Launcher for Plotly-based visualizations.

Example:
    >>> from logic.src.tracking.logging import plotting
    >>> plotting.plot_tsp(xy, tour, ax)
"""

from .attention import plot_attention_maps_wrapper
from .charts import plot_linechart
from .interactive import visualize_interactive_plot
from .routes import (
    discrete_cmap,
    draw_graph,
    plot_tsp,
    plot_vehicle_routes,
)

__all__ = [
    "plot_linechart",
    "draw_graph",
    "plot_tsp",
    "plot_vehicle_routes",
    "discrete_cmap",
    "plot_attention_maps_wrapper",
    "visualize_interactive_plot",
]
