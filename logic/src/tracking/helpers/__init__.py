"""Support utilities and helper functions for the WSTracker system.

This package contains miscellaneous utilities that assist in the integration
and operation of the tracking system, particularly focuses on the deep
learning training lifecycle with PyTorch Lightning.

Attributes:
    lightning_helpers: Utilities for hook management and visualization.

Example:
    >>> from logic.src.tracking.helpers import lightning_helpers
    >>> metrics = lightning_helpers.extract_metrics(cb_metrics, "train/")
"""
