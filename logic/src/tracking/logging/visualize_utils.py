"""Visualization utilities for the routing problems.

This module provides functions for visualizing routing solutions, graphs,
loss landscapes, and model training metrics. It primarily acts as a
re-export interface for the newer visualization sub-package.

DEPRECATED: Use logic.src.utils.logging.visualization package instead.

Attributes:
    plot_attention_heatmaps: Generates heatmaps of policy attention weights.
    plot_loss_landscape: Computes and visualizes the 2D loss surface.
    project_node_embeddings: Projects high-dimensional node features to 2D.
    log_weight_distributions: Records histograms of network parameters.

Example:
    >>> from logic.src.tracking.logging import visualize_utils
    >>> visualize_utils.plot_attention_heatmaps(policy, "heatmaps/")
"""

from logic.src.tracking.logging.visualization import (
    MyModelWrapper,
    get_batch,
    imitation_loss_fn,
    load_model_instance,
    log_weight_distributions,
    main,
    plot_attention_heatmaps,
    plot_logit_lens,
    plot_loss_landscape,
    plot_weight_trajectories,
    project_node_embeddings,
    rl_loss_fn,
    visualize_epoch,
)

__all__ = [
    "get_batch",
    "MyModelWrapper",
    "load_model_instance",
    "plot_weight_trajectories",
    "log_weight_distributions",
    "project_node_embeddings",
    "plot_attention_heatmaps",
    "plot_logit_lens",
    "imitation_loss_fn",
    "rl_loss_fn",
    "plot_loss_landscape",
    "visualize_epoch",
    "main",
]

if __name__ == "__main__":
    main()
