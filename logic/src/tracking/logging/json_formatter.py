"""
JSON Formatter for structured logging.
"""

import datetime
import json
import logging

from logic.src.interfaces import ITraversable


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON strings."""

    def format(self, record):
        """Format a log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, ITraversable):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)
