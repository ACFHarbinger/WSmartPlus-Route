"""Profiling utilities package.

Provides lightweight, WSTracker-integrated tools for measuring execution time,
memory usage, and throughput at multiple granularities:

* ExecutionProfiler / start_global_profiling / stop_global_profiling: sys.setprofile
  hook that records every project function call to a buffered CSV.
* BlockTimer / profile_block: context-manager timer for named code sections.
* MultiStepTimer: multi-phase timer that tracks sequential stages.
* profile_function: decorator that times every call to a function.
* MemorySnapshot: point-in-time GPU/CPU memory snapshot.
* MemoryTracker: background-thread memory monitor.
* ThroughputTracker: sliding-window items-per-second meter.
* ProfilingReport: reads and analyses an ExecutionProfiler CSV to produce
  top-function and per-module breakdowns.

All classes auto-forward results to the active WSTracker run via
contextlib.suppress-guarded calls — they are safe no-ops when no run is
active or optional dependencies are missing.

Attributes:
    ExecutionProfiler: sys.setprofile-based execution profiler.
    start_global_profiling: Function to start global execution profiling.
    stop_global_profiling: Function to stop global execution profiling.
    BlockTimer: Context-manager timer for code blocks.
    MultiStepTimer: Timer for multi-phase processes.
    profile_block: Helper function for block profiling.
    profile_function: Decorator for function profiling.
    MemorySnapshot: Record of memory usage at a specific time.
    MemoryTracker: Background monitor for memory usage.
    ThroughputTracker: Sliding-window throughput meter.
    ProfilingReport: Analyzer for execution profiler logs.

Example:
    >>> from logic.src.tracking.profiling import (
    ...     profile_block,
    ...     profile_function,
    ...     MemoryTracker,
    ...     ThroughputTracker,
    ... )
    >>> # Time a single block
    >>> with profile_block("data_loading", step=1):
    ...     dataset = load_data()
    >>> # Decorate a function
    >>> @profile_function(prefix="sim")
    ... def run_policy(obs):
    ...     return policy(obs)
    >>> # Track memory during training
    >>> with MemoryTracker(interval_sec=0.5, tag="train") as mem:
    ...     train_one_epoch()
    >>> mem.log_summary_to_run(step=1)
    >>> # Measure throughput
    >>> tracker = ThroughputTracker(unit="samples").start()
    >>> for batch in dataloader:
    ...     with tracker.step(len(batch)):
    ...         loss = train_step(batch)
    >>> tracker.log_to_run(step=1, prefix="train")
"""

from .memory import MemorySnapshot, MemoryTracker
from .profiler import ExecutionProfiler, start_global_profiling, stop_global_profiling
from .report import ProfilingReport
from .throughput import ThroughputTracker
from .timer import BlockTimer, MultiStepTimer, profile_block, profile_function

__all__ = [
    # Execution profiler (sys.setprofile)
    "ExecutionProfiler",
    "start_global_profiling",
    "stop_global_profiling",
    # Block / function timers
    "BlockTimer",
    "MultiStepTimer",
    "profile_block",
    "profile_function",
    # Memory profiling
    "MemorySnapshot",
    "MemoryTracker",
    # Throughput tracking
    "ThroughputTracker",
    # Report generation
    "ProfilingReport",
]
