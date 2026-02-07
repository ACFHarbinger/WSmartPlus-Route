"""
Baseline implementations for policy gradient methods.

This file acts as a facade for the baselines sub-package.
"""

from .baselines import (
    BASELINE_REGISTRY,
    Baseline,
    CriticBaseline,
    ExponentialBaseline,
    MeanBaseline,
    NoBaseline,
    POMOBaseline,
    RolloutBaseline,
    SharedBaseline,
    WarmupBaseline,
    get_baseline,
)

__all__ = [
    "Baseline",
    "MeanBaseline",
    "NoBaseline",
    "CriticBaseline",
    "SharedBaseline",
    "ExponentialBaseline",
    "POMOBaseline",
    "RolloutBaseline",
    "WarmupBaseline",
    "BASELINE_REGISTRY",
    "get_baseline",
]
