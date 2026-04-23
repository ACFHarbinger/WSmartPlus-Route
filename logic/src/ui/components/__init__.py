"""Reusable UI components for the dashboard.

This package provides a collection of interactive widgets and visualization
components for the Streamlit-based dashboard. It includes charts for
ALNS operator dynamics, attention mechanism heatmaps, benchmark
performance comparisons, and geographical map views.

Attributes:
    render_policy_viz: Renders visualization for routing policies.

Example:
    >>> from logic.src.ui.components import render_policy_viz
    >>> render_policy_viz(policy_name, metrics_dict)
"""

from .policy_viz import render_policy_viz

__all__ = ["render_policy_viz"]
