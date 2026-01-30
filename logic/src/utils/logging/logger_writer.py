import os
import sys
from datetime import datetime

import hydra


class LoggerWriter:
    """
    Fake file-like stream object that redirects writes to both terminal and file.
    """

    def __init__(self, terminal, filename):
        self.terminal = terminal
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        # Avoid writing tqdm progress bars (which use \r) to the file
        if "\r" not in message:
            self.log.write(message)
            self.log.flush()  # Ensure it writes immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


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
    sys.stdout = LoggerWriter(sys.stdout, log_file)
    sys.stderr = LoggerWriter(sys.stderr, log_file)

    print(f"Logging simulation output to: {log_file}")
    return log_file
