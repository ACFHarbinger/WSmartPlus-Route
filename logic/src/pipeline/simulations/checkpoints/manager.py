"""
Context manager for automatic checkpoint lifecycle management.

This module provides the checkpoint_manager context manager, which ensures
checkpoints are saved periodically and cleaned up or finalized on exit.

Attributes:
    CheckpointError: Custom exception for checkpoint-related failures.
    checkpoint_manager: Context manager for simulation state persistence.

Example:
    >>> # with checkpoint_manager(cp, 5, get_state) as hook:
    >>> #     for day in range(31): hook.on_day(day)
"""

from contextlib import contextmanager
from typing import Any, Callable, Optional

from .hooks import CheckpointHook


class CheckpointError(Exception):
    """Special exception to carry error results through the context manager

    Attributes:
        error_result: Dictionary containing error details and partial results.
    """

    def __init__(self, error_result: dict):
        """Initialize Class.

        Args:
            error_result: Partial simulation results and error trace.
        """
        self.error_result = error_result
        super().__init__(error_result["error"])


@contextmanager
def checkpoint_manager(
    checkpoint,
    checkpoint_interval: int,
    state_getter: Callable[[], Any],
    success_callback: Optional[Callable[[], None]] = None,
):
    """
    Context manager for automatic checkpoint lifecycle management.
    Args:
        checkpoint: SimulationCheckpoint instance.
        checkpoint_interval: Days between incremental checkpoints.
        state_getter: Callable that returns the current simulation state.
        success_callback: Optional function to call on successful completion.

    Yields:
        CheckpointHook: The hook object used to trigger daily checkpoints.
    """
    hook = CheckpointHook(checkpoint, checkpoint_interval, state_getter)
    try:
        yield hook
        hook.on_completion()
        if success_callback:
            success_callback()
    except Exception as e:
        error_result = hook.on_error(e)
        raise CheckpointError(error_result) from e
