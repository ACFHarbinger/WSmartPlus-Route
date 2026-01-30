import os
import sys
from datetime import datetime

import hydra


class LoggerWriter:
    """
    Fake file-like stream object that redirects writes to both terminal and file.
    """

    def __init__(self, terminal, filename, echo_to_terminal=True):
        self.terminal = terminal
        self.log = open(filename, "a", encoding="utf-8")
        self.echo_to_terminal = echo_to_terminal

    def write(self, message):
        if self.echo_to_terminal:
            self.terminal.write(message)

        # Avoid writing tqdm progress bars (which use \r) and cursor movements to the file
        # Filter ANSI escape codes for cursor up [A which might appear from tqdm
        if "\r" not in message and "\033[A" not in message and "[A" not in message:
            self.log.write(message)
            self.log.flush()  # Ensure it writes immediately

    def flush(self):
        if self.echo_to_terminal:
            self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: -1)()

    def __getattr__(self, name):
        """Delegate any other attribute access to the terminal stream."""
        return getattr(self.terminal, name)


def setup_logger_redirection():
    """Redirects stdout and stderr to a timestamped log file."""
    try:
        base_dir = hydra.utils.get_original_cwd()
    except (ValueError, ImportError):
        base_dir = os.getcwd()

    log_dir = os.path.join(base_dir, "logs/simulations")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    # Tee stdout and stderr
    # stdout goes ONLY to log (silent_terminal=True -> echo_to_terminal=False)
    sys.stdout = LoggerWriter(sys.stdout, log_file, echo_to_terminal=False)

    # stderr goes to BOTH (so we see tqdm), but LoggerWriter.write filters tqdm from file
    sys.stderr = LoggerWriter(sys.stderr, log_file, echo_to_terminal=True)

    # Print to ORIGINAL stdout so user knows where logs are, before redirection takes full effect
    # (Though we just redirected stdout, so this will go to log only if we use print)
    # let's write to the original stderr or stdout if we want user to see it
    sys.stderr.terminal.write(f"Logging simulation output to: {log_file}\n")

    return log_file
