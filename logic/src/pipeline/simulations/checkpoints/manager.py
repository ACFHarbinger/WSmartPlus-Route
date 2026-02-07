"""
Context manager for automatic checkpoint lifecycle management.
"""

from contextlib import contextmanager
from typing import Any, Callable, Optional

from .hooks import CheckpointHook


class CheckpointError(Exception):
    """Special exception to carry error results through the context manager"""

    def __init__(self, error_result: dict):
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
