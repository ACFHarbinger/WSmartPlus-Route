"""
Policy naming and configuration constants.
"""

# Policies with engine options
ENGINE_POLICIES = {
    "vrpp": ["gurobi", "hexaly"],
}

# Policies that parse threshold from string
THRESHOLD_POLICIES = ["vrpp", "sans", "hgs", "alns", "bcp"]

# Policies with special config chars (e.g., lac_a_1.0, lac_b_2.0)
CONFIG_CHAR_POLICIES = {"lac": ["a", "b"]}

# Simple name mappings (no threshold parsing)
SIMPLE_POLICIES = {
    ("am", "ddam", "transgcn"): "neural",
    ("last_minute",): "last_minute",
    ("regular",): "regular",
    ("bcp",): "bcp",
    ("lkh",): "lkh",
    ("tsp",): "tsp",
    ("cvrp",): "cvrp",
}
