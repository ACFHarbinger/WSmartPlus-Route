"""Database management and analytics for WSTracker.

This package provides the operational toolset for managing the persistent SQLite
tracking store. It includes high-level commands for database cleanup,
maintenance, and inspection, as well as analytics helpers for generating
statistical summaries and metric reports.

Primarily used via the python -m logic.src.tracking.database CLI.

Attributes:
    clean_database: Resets the database to an empty state.
    compact_database: Performs maintenance and reclaims disk space.
    export_run: Serializes a run record to JSON.
    inspect_database: Provides a high-level status of the database.
    metrics_summary: Generates reports on metric evolution.
    prune_database: Removes old or irrelevant runs.
    stats_database: Generates a comprehensive statistical dashboard.

Example:
    >>> from logic.src.tracking.database import inspect_database
    >>> inspect_database()
"""

from .cmd_stats import metrics_summary, stats_database
from .commands import (
    clean_database,
    compact_database,
    export_run,
    inspect_database,
    prune_database,
)

__all__ = [
    "clean_database",
    "compact_database",
    "export_run",
    "inspect_database",
    "metrics_summary",
    "prune_database",
    "stats_database",
]
