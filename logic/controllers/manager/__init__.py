"""
Batch Manager package.

This package provides the ``BatchManager`` and supporting classes for running
multiple WSmart-Route simulations (or training/evaluation runs) in sequence with
pre/post-processing hooks and global setup/teardown lifecycle steps.

Public API
----------
``BatchManager``
    Top-level orchestrator.  Load from a YAML config via
    ``BatchManager.from_yaml(path)`` then call ``.run()``.

``BatchJob``
    Dataclass representing a single run (task + Hydra overrides + hooks).

``JobResult``
    Outcome of a single job execution.

``BatchStep``
    A single hook step (setup, teardown, pre-step, or post-step).

``BatchExpander``
    Expands a YAML ``expand`` block into a Cartesian product of ``BatchJob``
    instances.

``BatchExecutor``
    Executes an ordered list of ``BatchJob`` objects.

``run_step``
    Low-level dispatch function for executing a ``BatchStep``.

``STEP_HANDLERS``
    Registry mapping step type strings to handler functions.  Add entries here
    to register new step types without modifying core files.
"""

from .batch_executor import BatchExecutor
from .batch_expander import BUILTIN_DIM_OVERRIDES, BatchExpander
from .batch_job import BatchJob, JobResult
from .batch_manager import BatchManager
from .batch_step import CONDITION_ALL_SUCCEEDED, CONDITION_ALWAYS, CONDITION_ANY_FAILED, BatchStep
from .batch_step_executor import STEP_HANDLERS, run_step

__all__ = [
    "BatchManager",
    "BatchExecutor",
    "BatchExpander",
    "BatchJob",
    "JobResult",
    "BatchStep",
    "run_step",
    "STEP_HANDLERS",
    "BUILTIN_DIM_OVERRIDES",
    "CONDITION_ALWAYS",
    "CONDITION_ALL_SUCCEEDED",
    "CONDITION_ANY_FAILED",
]
