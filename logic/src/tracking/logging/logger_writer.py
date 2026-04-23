"""Logger redirection to file for simulation output.

Provides LoggerWriter to tee stdout/stderr to both terminal and file,
filtering tqdm progress bars from file output.

Attributes:
    LoggerWriter: File-like object that redirects stream output.
    setup_logger_redirection: Configures sys.stdout/stderr redirection.

Example:
    >>> import sys
    >>> with open("output.log", "w") as f:
    ...     writer = LoggerWriter(sys.stdout, f)
    ...     writer.write("This goes to both screen and file")
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional, TextIO

import hydra
from loguru import logger


class LoggerWriter:
    """Fake file-like stream object that redirects writes to both terminal and file.

    Useful for capture-and-tee scenarios where output must be preserved in a
    persistent log file while remaining visible in the interactive terminal.
    Automatically strips ANSI escape codes and filters out carriage-return
    lines (commonly used by tqdm) to keep text logs clean.

    Attributes:
        terminal (TextIO): The original stream being tee-ed.
        filename (str): Path to the output log file.
        echo_to_terminal (bool): Whether writes are still echoed to screen.
        log (TextIO): File handle for the persistence log.
    """

    def __init__(self, terminal: TextIO, filename: str, echo_to_terminal: bool = True) -> None:
        """Initialize the logger writer.

        Args:
            terminal: Original terminal stream (stdout or stderr).
            filename: Path to the log file.
            echo_to_terminal: If True, also write to terminal. Defaults to True.
        """
        self.terminal = terminal
        self.filename = filename
        self.echo_to_terminal = echo_to_terminal
        # Open file in append mode.
        self.log = open(filename, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, message: str) -> None:
        """Write message to terminal and/or log file.

        Strips ANSI escape sequences and filters out lines containing carriage
        returns before writing to the log file.

        Args:
            message: The raw text string to write.
        """
        if self.echo_to_terminal:
            self.terminal.write(message)

        # Strip ANSI escape sequences for the log file
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_message = ansi_escape.sub("", message)

        # Skip lines containing carriage returns (progress bars)
        # We split using keepends=True handled by splitlines to strictly evaluate line-by-line
        # ensuring that logs don't merge even when mixed with progress updates.
        for line in clean_message.splitlines(keepends=True):
            if "\r" not in line:
                self.log.write(line)

        self.log.flush()  # Ensure it writes immediately

    def flush(self) -> None:
        """Flush both terminal and log file buffers."""
        if self.echo_to_terminal:
            self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        """Close the log file handle."""
        self.log.close()

    def isatty(self) -> bool:
        """Return whether the terminal is a TTY.

        Returns:
            bool: True if the underlying terminal is a TTY.
        """
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self) -> int:
        """Return the file descriptor number of the terminal.

        Returns:
            int: File descriptor index, or -1 if unavailable.
        """
        return getattr(self.terminal, "fileno", lambda: -1)()

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attribute access to the terminal stream.

        Args:
            name: Attribute name.

        Returns:
            Any: The attribute from the original terminal stream.
        """
        return getattr(self.terminal, name)


def setup_logger_redirection(
    log_file: Optional[str] = None,
    silent: bool = False,
    redirect_stdout: bool = True,
    redirect_stderr: bool = True,
    echo_to_terminal: bool = False,
) -> str:
    """Redirects stdout and stderr to a timestamped log file.

    Automatically handles log directory creation and reconfigures Loguru
    to respect the redirected stderr stream.

    Args:
        log_file: Optional path to an existing log file to reuse.
            Defaults to None.
        silent: If True, do not print the redirection announcement.
            Defaults to False.
        redirect_stdout: If True, redirect sys.stdout. Defaults to True.
        redirect_stderr: If True, redirect sys.stderr. Defaults to True.
        echo_to_terminal: If True, keep echoing to the original terminal.
            Defaults to False.

    Returns:
        str: Absolute path to the log file used.
    """
    if log_file is None:
        try:
            base_dir = hydra.utils.get_original_cwd()
        except (ValueError, ImportError):
            base_dir = os.getcwd()

        log_dir = os.path.join(base_dir, "logs/simulations")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{timestamp}.log")

    # Tee stdout and stderr based on flags
    if redirect_stdout:
        if not isinstance(sys.stdout, LoggerWriter):
            sys.stdout = LoggerWriter(sys.stdout, log_file, echo_to_terminal=echo_to_terminal)
        else:
            sys.stdout.echo_to_terminal = echo_to_terminal
    if redirect_stderr:
        if not isinstance(sys.stderr, LoggerWriter):
            sys.stderr = LoggerWriter(sys.stderr, log_file, echo_to_terminal=echo_to_terminal)
        else:
            sys.stderr.echo_to_terminal = echo_to_terminal

    # Reconfigure Loguru to use the redirected sys.stderr
    with contextlib.suppress(ImportError):
        logger.remove()
        logger.add(sys.stderr, format="[{level}] {message}", colorize=False)

    # Print to ORIGINAL stderr so user knows where logs are, before redirection takes full effect
    if not silent:
        if hasattr(sys.stderr, "terminal"):
            sys.stderr.terminal.write(f"Logging simulation output to: {log_file}\n")
        else:
            sys.stderr.write(f"Logging simulation output to: {log_file}\n")

    return log_file
