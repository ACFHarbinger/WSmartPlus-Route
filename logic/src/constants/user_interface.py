"""
Constants for simulation progress bar display.

Attributes:
    PBAR_WAIT_TIME: Progress bar update interval (seconds)
    TQDM_COLOURS: TQDM progress bar color cycling for parallel workers
"""

from typing import List

# Progress bar update interval (seconds)
# Controls tqdm refresh rate for multi-core simulation workers.
PBAR_WAIT_TIME: float = 0.1  # seconds (100ms refresh interval)

# TQDM progress bar color cycling
# Used in: Parallel simulation workers (one color per worker)
TQDM_COLOURS: List[str] = [
    "red",
    "blue",
    "green",
    "yellow",
    "magenta",
    "cyan",
]
