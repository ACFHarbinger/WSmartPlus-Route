"""JSON Formatter for structured logging.

Provides a logging formatter that serializes log records into JSON strings,
suitable for consumption by ELK, Splunk, or other log aggregators.

Attributes:
    JsonFormatter: A logging.Formatter subclass for JSON output.

Example:
    >>> import logging
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(JsonFormatter())
    >>> logger = logging.getLogger("json")
    >>> logger.addHandler(handler)
"""

from __future__ import annotations

import datetime
import json
import logging

from logic.src.interfaces import ITraversable


class JsonFormatter(logging.Formatter):
    """Formatter that outputs log records as structured JSON strings.

    Extracts standard record attributes (timestamp, level, message, module, etc.)
    and incorporates optional 'extra_fields' if they implement ITraversable.

    Attributes:
        No public attributes beyond standard logging.Formatter ones.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The logging record to format.

        Returns:
            str: The JSON-serialized log entry.
        """
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        extra: object = getattr(record, "extra_fields", None)
        if extra is not None and isinstance(extra, ITraversable):
            log_entry.update(extra)

        return json.dumps(log_entry)
