"""Core tracking primitives for the WSmart-Route experiment tracking system.

This package provides the fundamental building blocks for experiment tracking,
including the central tracker coordinator, individual run lifecycle
management, and the persistent SQLite storage backend.

Attributes:
    Tracker: Central coordinator for experiment tracking.
    get_tracker: Retrieves the process-global tracker instance.
    Run: Main class for interacting with an individual tracked run.
    get_active_run: Retrieves the current active run instance.
    set_active_run: Registers a run instance as the active one.
    TrackingStore: Persistent SQLite store for experiment tracking data.

Example:
    >>> import logic.src.tracking.core as core
    >>> tracker = core.Tracker(tracking_uri="./mlruns")
    >>> with tracker.start_run("VRPP-Baseline") as run:
    ...     run.log_metric("accuracy", 0.95)
"""

from .run import Run, get_active_run, set_active_run
from .store import TrackingStore
from .tracker import Tracker, get_tracker

__all__ = [
    "Tracker",
    "get_tracker",
    "Run",
    "get_active_run",
    "set_active_run",
    "TrackingStore",
]
