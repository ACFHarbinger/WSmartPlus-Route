"""
Model configuration constants.
"""
from typing import List

# Model configurations
SUB_NET_ENCS: List[str] = ["tgc"]
PRED_ENC_MODELS: List[str] = ["tam"]
ENC_DEC_MODELS: List[str] = ["ddam"]

# Dimension constants
NODE_DIM: int = 3  # Coordinate (2) + Demand/Value (1)
STATIC_DIM: int = 2  # Coordinate (x, y)
DEPOT_DIM: int = 2  # Coordinate (x, y)

# Step context dimensionality offsets
WC_STEP_CONTEXT_OFFSET: int = 2
VRPP_STEP_CONTEXT_OFFSET: int = 1

# Temporal defaults
DEFAULT_TEMPORAL_HORIZON: int = 10

# Architecture Constants
TANH_CLIPPING: float = 10.0
NORM_EPSILON: float = 1e-5
FEED_FORWARD_EXPANSION: int = 4
