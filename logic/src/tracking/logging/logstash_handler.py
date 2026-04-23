"""Logstash TCP Handler for remote logging.

Provides a logging handler that transmits structured JSON log entries over
a TCP socket to a remote Logstash instance.

Attributes:
    LogstashTcpHandler: A logging.Handler subclass for remote TCP logging.

Example:
    >>> handler = LogstashTcpHandler(host="localhost", port=5000)
    >>> logger = logging.getLogger("remote")
    >>> logger.addHandler(handler)
"""

from __future__ import annotations

import logging
import socket
from typing import Any

from .json_formatter import JsonFormatter


class LogstashTcpHandler(logging.Handler):
    """Handler that sends JSON logs over TCP to Logstash.

    Transmits formatted log records as newline-terminated strings to a
    Logstash listener.

    Attributes:
        host (str): Logstash server hostname.
        port (int): Logstash server port.
    """

    def __init__(self, host: str = "localhost", port: int = 5000) -> None:
        """Initialize the TCP handler for Logstash.

        Args:
            host: Logstash server hostname. Defaults to "localhost".
            port: Logstash server port. Defaults to 5000.
        """
        super().__init__()
        self.host = host
        self.port = port
        self.formatter = JsonFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by sending it to Logstash over TCP.

        Args:
            record: The logging record to emit.
        """
        try:
            msg = self.format(record) + "\n"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.host, self.port))
                s.sendall(msg.encode("utf-8"))
        except Exception:
            # Silent failure to avoid breaking the application
            pass
