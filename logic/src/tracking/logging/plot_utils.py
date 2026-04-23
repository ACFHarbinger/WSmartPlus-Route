"""Plotting utilities for the WSmart+ Route framework.

This file acts as a facade for the plotting sub-package, providing a central
interface for generating static and interactive visualizations of routing
solutions, attention maps, and training metrics.

Attributes:
    plot_linechart: Generates 2D line charts for metric tracking.
    draw_graph: Visualizes problem graphs and nodes.
    plot_tsp: Plots Traveling Salesman Problem solutions.
    plot_vehicle_routes: Visualizes multi-vehicle routing paths.
    plot_attention_maps_wrapper: Batch plotting of attention mechanism weights.
    visualize_interactive_plot: Launcher for web-based interactive charts.

Example:
    >>> from logic.src.tracking.logging import plot_utils
    >>> plot_utils.plot_linechart(data, "output.png")
"""

import matplotlib.pyplot as plt

from .plotting import (
    discrete_cmap,
    draw_graph,
    plot_attention_maps_wrapper,
    plot_linechart,
    plot_tsp,
    plot_vehicle_routes,
    visualize_interactive_plot,
)

__all__ = [
    "plot_linechart",
    "draw_graph",
    "plot_tsp",
    "plot_vehicle_routes",
    "discrete_cmap",
    "plot_attention_maps_wrapper",
    "visualize_interactive_plot",
    "plt",
]
