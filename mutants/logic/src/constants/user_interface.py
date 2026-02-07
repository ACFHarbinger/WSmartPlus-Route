"""
Constants related to the User Interface and Visualization.
"""
from typing import List, Tuple, Union

# WSmart+ route simulation settings (Visuals)
PBAR_WAIT_TIME: float = 0.1

# Correcting TQDM_COLOURS to match original exactly
TQDM_COLOURS = [
    "red",
    "blue",
    "green",
    "yellow",
    "magenta",
    "cyan",
]

# Plotting
MARKERS: List[str] = ["P", "s", "^", "8", "*"]
PLOT_COLOURS: List[str] = [
    "red",
    "blue",
    "green",
    "yellow",
    "magenta",
    "cyan",
    "black",
]
LINESTYLES: List[Union[str, Tuple[int, Tuple[int, ...]]]] = [
    "dotted",
    "dashed",
    "dashdot",
    (0, (3, 5, 1, 5, 1, 5)),
    "solid",
]

# GUI settings
CTRL_C_TIMEOUT: float = 2.0

APP_STYLES: List[str] = ["fusion", "windows", "windowsxp", "macintosh"]
