"""
Constants for look_ahead_aux module.
"""

# Collection Parameters
COLLECTION_TIME_MINUTES = 3.0  # Time to collect one bin
VEHICLE_SPEED_KMH = 40.0  # Average vehicle speed in km/h

# Optimization Penalties
PENALTY_MUST_GO_MISSED = 10000.0  # Penalty for missing a mandatory bin

# Capacity
MAX_CAPACITY_PERCENT = 100.0  # Maximum bin capacity percentage

# Gurobi Parameters
MIP_GAP = 0.01
HEURISTICS_RATIO = 0.5
NODEFILE_START_GB = 0.5

# LAC Constants
DEFAULT_SHIFT_DURATION = 390
DEFAULT_V_VALUE = 1.0
DEFAULT_COMBINATION = [500, 75, 0.95, 0, 0.095, 0, 0]
DEFAULT_TIME_LIMIT = 600
