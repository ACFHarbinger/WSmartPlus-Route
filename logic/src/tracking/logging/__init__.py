"""Logging and diagnostic output utilities for WSmart-Route.

This package provides tools for standardizing console output, file logging,
and structured telemetry across both local development and distributed
cluster environments.

Attributes:
    get_pylogger: Factory for rank-zero-safe Python loggers.

Example:
    >>> from logic.src.tracking.logging import get_pylogger
    >>> logger = get_pylogger("my_module")
"""

from logic.src.tracking.logging.pylogger import get_pylogger

__all__ = ["get_pylogger"]
