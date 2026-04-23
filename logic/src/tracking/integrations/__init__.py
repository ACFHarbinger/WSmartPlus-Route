"""Pipeline integration helpers for the WSTracker ecosystem.

This package provides high-level trackers for various stages of the machine
learning pipeline, including in-memory data mutation tracking, filesystem
lineage, PyTorch Lightning callbacks, and simulation-specific instrumentation.

Attributes:
    TrackingCallback: PyTorch Lightning callback for run instrumentation.
    SimulationRunTracker: Specialized tracker for multi-day simulation episodes.
    get_sim_tracker: Retrieves the process-global simulation tracker instance.
    RuntimeDataTracker: Tracks in-memory dataset distribution mutations.
    FilesystemTracker: Tracks filesystem-level data events and lineage.
    ZenMLBridge: Integration bridge for ZenML pipeline tracking.

Example:
    >>> import logic.src.tracking.integrations as integrations
    >>> data_tracker = integrations.RuntimeDataTracker(run)
    >>> fs_tracker = integrations.FilesystemTracker(run)
"""

from .data import RuntimeDataTracker
from .filesystem import FilesystemTracker
from .lightning import TrackingCallback
from .simulation import SimulationRunTracker, get_sim_tracker
from .zenml_bridge import ZenMLBridge

__all__ = [
    "TrackingCallback",
    "SimulationRunTracker",
    "get_sim_tracker",
    "RuntimeDataTracker",
    "FilesystemTracker",
    "ZenMLBridge",
]
