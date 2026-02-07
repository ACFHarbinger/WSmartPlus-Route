"""
Simulation and Physics constants.
"""
from typing import Dict, List

# Distance matrix
EARTH_RADIUS: int = 6371
EARTH_WMP_RADIUS: int = 6378137

# WSmart+ route simulation metrics
METRICS: List[str] = [
    "overflows",
    "kg",
    "ncol",
    "kg_lost",
    "km",
    "kg/km",
    "cost",
    "profit",
]
SIM_METRICS: List[str] = METRICS + ["days", "time"]
DAY_METRICS: List[str] = ["day"] + METRICS + ["tour"]
LOSS_KEYS: List[str] = ["nll", "reinforce_loss", "baseline_loss"]

# Problem definition
MAX_WASTE: float = 1.0
MAX_LENGTHS: Dict[int, int] = {20: 2, 50: 3, 100: 4, 150: 5, 225: 6, 317: 7}
VEHICLE_CAPACITY: float = 100.0

PROBLEMS: List[str] = [
    "vrpp",
    "cvrpp",
    "wcvrp",
    "cwcvrp",
    "sdwcvrp",
    "scwcvrp",
]
