"""Shared database connection utilities for tracking database commands.

This module provides the common database connection factory and default
storage path used by the tracking CLI utilities. It ensures consistent
connection parameters (timeouts, row factories) across all management
subcommands.

Attributes:
    DB_PATH: Default relative path to the WSTracker SQLite database.

Example:
    >>> from logic.src.tracking.database.shared import _conn
    >>> conn = _conn()
    >>> conn.execute("SELECT COUNT(*) FROM runs").fetchone()
"""

from __future__ import annotations

import sqlite3

DB_PATH: str = "assets/tracking/tracking.db"


def _conn(timeout: float = 5.0) -> sqlite3.Connection:
    """Creates a short-lived SQLite connection for CLI operations.

    Args:
        timeout: Maximum time in seconds to wait for a database lock.
            Defaults to 5.0.

    Returns:
        sqlite3.Connection: A configured connection with dictionary-like row access.
    """
    c = sqlite3.connect(DB_PATH, timeout=timeout)
    c.row_factory = sqlite3.Row
    return c
