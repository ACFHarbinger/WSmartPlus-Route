"""
Statistical functions and constants.
"""

import statistics
from typing import Any, Callable, Dict

STATS_FUNCTION_MAP: Dict[str, Callable[..., Any]] = {
    "mean": statistics.mean,
    "stdev": statistics.stdev,
    "median": statistics.median,
    "mode": statistics.mode,
    "var": statistics.variance,
    "quant": statistics.quantiles,
    "size": len,
    "sum": sum,
    "min": min,
    "max": max,
}
