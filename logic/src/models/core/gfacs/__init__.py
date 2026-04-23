"""GFACS: GFlowNet Ant Colony System.

This package implements the GFACS architecture for neural routing. It
combines ACO construction with GFlowNet Trajectory Balance training.

Attributes:
    GFACS: Unified training wrapper.
    GFACSPolicy: Neural GFlowNet with ACO construction.

Example:
    >>> from logic.src.models.core.gfacs import GFACS
"""

from .model import GFACS as GFACS
from .policy import GFACSPolicy as GFACSPolicy

__all__ = ["GFACS", "GFACSPolicy"]
