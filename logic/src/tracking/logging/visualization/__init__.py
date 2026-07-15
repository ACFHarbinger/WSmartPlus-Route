"""Visualization helpers for experiment tracking (§A.2)."""

from logic.src.tracking.logging.visualization.heatmaps import (
    capture_runtime_attention,
    log_attention_heatmaps_to_backends,
    maybe_log_eval_attention_heatmaps,
    plot_attention_heatmaps,
)

__all__ = [
    "capture_runtime_attention",
    "log_attention_heatmaps_to_backends",
    "maybe_log_eval_attention_heatmaps",
    "plot_attention_heatmaps",
]
