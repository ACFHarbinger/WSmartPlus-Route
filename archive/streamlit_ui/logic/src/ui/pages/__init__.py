"""Streamlit dashboard page entry points.

This package exports the main rendering functions for each dashboard mode,
allowing for dynamic layout switching in the core app orchestrator.

Example:
    from logic.src.ui.pages import render_algorithms
    render_algorithms()

Attributes:
    render_training_monitor: Real-time DRL training metrics.
    render_simulation_visualizer: Interactive tour/bin visualization.
    render_simulation_summary: KPI and KPI comparison dashboard.
    render_benchmark_analysis: Cross-hardware solver analysis.
    render_data_explorer: Interactive dataset profiling.
    render_experiment_tracker: Multi-backend run auditing.
    render_hpo_tracker: Optuna study visualization.
    render_algorithms: Global registry discovery interface.
"""

from .algorithms import render_algorithms
from .benchmark import render_benchmark_analysis
from .data_explorer import render_data_explorer
from .experiment_tracker import render_experiment_tracker
from .hpo_tracker import render_hpo_tracker
from .simulation import render_simulation_summary, render_simulation_visualizer
from .training import render_training_monitor

__all__ = [
    "render_training_monitor",
    "render_simulation_visualizer",
    "render_simulation_summary",
    "render_benchmark_analysis",
    "render_data_explorer",
    "render_experiment_tracker",
    "render_hpo_tracker",
    "render_algorithms",
]
