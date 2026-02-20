"""
Global Execution Profiler

Provides a `sys.setprofile` hook to record execution times for all functions
called within the project, logging them to a CSV format file.
"""

import atexit
import logging
import os
import sys
import threading
import time
from typing import Any, Optional

from logic.src.constants import ROOT_DIR

logger = logging.getLogger(__name__)

_ROOT_DIR = str(ROOT_DIR)

# Standard library and third party directories to filter out
_LIB_DIRS = (
    "site-packages",
    "dist-packages",
    "lib/python",
    "<frozen",
    "<built-in",
    "logic/src/utils/logging/profiler.py",  # Filter self
)


class ExecutionProfiler:
    """Tracks and logs function execution times globally."""

    def __init__(self, log_dir: str = "logs"):
        """Initialize the profiler."""
        self.log_dir = os.path.join(_ROOT_DIR, log_dir) if not os.path.isabs(log_dir) else log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"function_execution_times_{timestamp}.csv")
        self.active = False
        self._lock = threading.Lock()

        # Thread-local storage for tracking call stacks and start times
        self._local = threading.local()

        # Write CSV header but defer opening file continuously
        try:
            with open(self.log_file, "w") as f:
                f.write("timestamp,file,class,function,duration_sec\n")
        except Exception as e:
            logger.error(f"Failed to initialize profiler log file: {e}")

    def _should_profile(self, filename: str) -> bool:
        """Determines if a file should be profiled (within project)."""
        if not filename or filename == "<string>":
            return False

        # Fast checks to ignore built-ins and standard libs
        for lib_dir in _LIB_DIRS:
            if lib_dir in filename:
                return False

        # Check if it's within the project root
        return filename.startswith(_ROOT_DIR)

    def _get_class_name(self, frame: Any) -> str:
        """Attempts to extract the class name from a frame's locals."""
        # Check for 'self' (instance method)
        if "self" in frame.f_locals:
            return frame.f_locals["self"].__class__.__name__

        # Check for 'cls' (class method)
        if "cls" in frame.f_locals:
            cls = frame.f_locals["cls"]
            if hasattr(cls, "__name__"):
                return cls.__name__

        return ""

    def profile_hook(self, frame: Any, event: str, arg: Any) -> None:
        """The main hook executed by sys.setprofile."""
        if not self.active:
            return

        # Initialize thread-local call stack if not exists
        if not hasattr(self._local, "call_stack"):
            self._local.call_stack = []

        code = frame.f_code
        filename = code.co_filename

        if not self._should_profile(filename):
            return

        func_name = code.co_name

        if event == "call":
            # Record start time
            start_time = time.perf_counter()
            class_name = self._get_class_name(frame)

            # Using id(frame) as a unique identifier for the specific call
            self._local.call_stack.append((id(frame), start_time, filename, class_name, func_name))

        elif event == "return":
            # Find the matching call in the stack
            frame_id = id(frame)

            # Search backwards for the matching call
            # (handles recursive calls and unexpected stack changes)
            match_index = -1
            for i in range(len(self._local.call_stack) - 1, -1, -1):
                if self._local.call_stack[i][0] == frame_id:
                    match_index = i
                    break

            if match_index != -1:
                # Extract call data
                _, start_time, filename, class_name, func_name = self._local.call_stack[match_index]

                # Cleanup stack properly (remove this and any orphaned calls above it)
                self._local.call_stack = self._local.call_stack[:match_index]

                duration = time.perf_counter() - start_time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Relative filename for cleaner logging
                rel_filename = os.path.relpath(filename, _ROOT_DIR)

                log_entry = f"{timestamp},{rel_filename},{class_name},{func_name},{duration:.6f}\n"

                # Write to file (using lock for thread safety if applied generically)
                with self._lock:
                    try:
                        with open(self.log_file, "a") as f:
                            f.write(log_entry)
                    except Exception:
                        pass  # Fail silently during high-volume profiling

    def start(self) -> None:
        """Starts profiling globally."""
        if not self.active:
            self.active = True
            # Set profile for current thread
            sys.setprofile(self.profile_hook)
            # Instruct threading module to set profile for all new threads
            threading.setprofile(self.profile_hook)
            logger.info(f"Global execution profiling started. Logging to {self.log_file}")

    def stop(self) -> None:
        """Stops profiling."""
        if self.active:
            self.active = False
            sys.setprofile(None)
            threading.setprofile(None)
            logger.info("Global execution profiling stopped.")


# Singleton instance
_profiler_instance: Optional[ExecutionProfiler] = None


def start_global_profiling(log_dir: str = "logs") -> None:
    """Initialize and start the global profiling hook.

    Args:
        log_dir: Directory (relative to ROOT_DIR or absolute) for the CSV output file.
    """
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = ExecutionProfiler(log_dir=log_dir)
        _profiler_instance.start()
        # Ensure profiling is stopped and resources released on exit
        atexit.register(stop_global_profiling)


def stop_global_profiling() -> None:
    """Stop the global profiling hook."""
    global _profiler_instance
    if _profiler_instance is not None:
        _profiler_instance.stop()
