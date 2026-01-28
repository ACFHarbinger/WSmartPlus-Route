"""
Structured JSON logging utilities for ELK stack integration.

Provides handlers and formatters to output logs in JSON format,
optionally sending them via TCP to Logstash.
"""

import datetime
import json
import logging
import socket
from typing import Any, Optional


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON strings."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class LogstashTcpHandler(logging.Handler):
    """Handler that sends JSON logs over TCP to Logstash."""

    def __init__(self, host: str = "localhost", port: int = 5000):
        super().__init__()
        self.host = host
        self.port = port
        self.formatter = JsonFormatter()

    def emit(self, record):
        try:
            msg = self.format(record) + "\n"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.host, self.port))
                s.sendall(msg.encode("utf-8"))
        except Exception:
            # Silent failure to avoid breaking the application
            pass


def get_structured_logger(
    name: str = "wsmart.structured", level: int = logging.INFO, logstash_host: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger configured for structured JSON output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler for local debugging
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        logger.addHandler(console)

        # Logstash handler if host provided
        if logstash_host:
            ls_handler = LogstashTcpHandler(host=logstash_host)
            logger.addHandler(ls_handler)

    return logger


def log_test_metric(name: str, value: Any, logger_name: str = "wsmart.structured"):
    """Convenience function to log a test metric."""
    logger = logging.getLogger(logger_name)
    extra = {"type": "test_metric", "metric_name": name, "metric_value": value}
    logger.info(f"Metric: {name}={value}", extra={"extra_fields": extra})
