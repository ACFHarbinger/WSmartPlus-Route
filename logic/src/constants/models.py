"""
Model configuration constants.

Problem Size Naming Conventions
-------------------------------
Three variable names describe "number of nodes" across the codebase.
Each carries distinct semantics — do NOT rename one to another:

    num_loc     Customer locations, **excluding** the depot.
                Used in: configs (EnvConfig), generators, environments, CLI.
                Example: ``num_loc = 50`` → 50 customers.

    graph_size  Total nodes in the tensor representation, **including** the depot.
                Used in: model forward passes (local variable unpacked from tensor shape),
                encoder/decoder comments, simulation context.
                Relationship: ``graph_size = num_loc + 1`` for VRP problems.

    n_nodes     Customer nodes inside classical solvers, **excluding** the depot.
                Used in: HGS, ALNS, BCP (local/instance variables).
                Typically computed as ``len(dist_matrix) - 1``.

Propagation flow:
    Config (num_loc) → Generator (num_loc) → Environment (num_loc)
        → Model tensors (graph_size = num_loc + 1)
        → Classical policies (n_nodes = num_loc)
        → Simulator context (graph_size)
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

DEFAULT_MOE_KWARGS = {
    "encoder": {
        "hidden_act": "ReLU",
        "num_experts": 4,
        "k": 2,
        "noisy_gating": True,
    },
    "decoder": {
        "light_version": True,
        "num_experts": 4,
        "k": 2,
        "noisy_gating": True,
    },
}
