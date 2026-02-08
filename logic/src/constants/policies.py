"""
Policy naming and configuration constants.

This module defines policy classification registries for the configuration
and instantiation system. Used by:
- logic/src/policies/__init__.py (policy adapter factory)
- logic/src/configs/policies/*.py (config dataclass selection)
- logic/src/cli/ts_parser.py (command-line policy parsing)

Policy Classification System
-----------------------------
Policies are categorized into 4 groups based on their configuration needs:

1. **ENGINE_POLICIES**: Solvers with multiple backend engines
   - Require engine selection parameter (e.g., gurobi vs hexaly)
   - Example: vrpp policy can use Gurobi (exact MIP) or Hexaly (local search)

2. **THRESHOLD_POLICIES**: Algorithms with tunable threshold parameters
   - Parse numeric suffix from policy name (e.g., "sans_0.5" → threshold=0.5)
   - Example: "hgs_1000" → HGS with 1000 iterations
   - Used for hyperparameter sweeps via naming convention

3. **CONFIG_CHAR_POLICIES**: Policies with variant suffixes
   - Use alphabetic suffix for configuration selection (e.g., "lac_a", "lac_b")
   - Example: "lac_a_1.0" → LAC engine 'a' with threshold 1.0
   - Enables multiple configurations of same base algorithm

4. **SIMPLE_POLICIES**: Fixed-configuration policies with direct name mapping
   - No parsing required; display name → config file name
   - Example: "am" → neural, "lkh" → lkh
   - Most common type; used for standard baseline policies

Configuration File Resolution
------------------------------
For a policy named "sans_0.95":
1. Check THRESHOLD_POLICIES → "sans" found
2. Parse threshold: 0.95
3. Load: assets/configs/policies/policy_sans.yaml
4. Pass threshold to PolicyConfig constructor

For a policy named "last_minute":
1. Check SIMPLE_POLICIES → ("last_minute",) found
2. Load: assets/configs/policies/policy_last_minute.yaml
3. No threshold parsing needed

When to Use Each Type
---------------------
- Use ENGINE_POLICIES if your solver has swappable backends (e.g., Gurobi/CPLEX/Hexaly)
- Use THRESHOLD_POLICIES if tuning a single numeric parameter is common
- Use CONFIG_CHAR_POLICIES if you need named variants (e.g., conservative vs aggressive)
- Use SIMPLE_POLICIES for everything else (no dynamic parameters)
"""

# Policies with engine options
# Maps policy name → list of available solver backends.
# Use when: Policy supports multiple exact/metaheuristic engines for same problem.
# CLI syntax: --policy vrpp --engine gurobi
ENGINE_POLICIES = {
    "vrpp": ["gurobi", "hexaly"],  # Vehicle Routing with Profits: Gurobi (exact MIP) or Hexaly (local search)
}

# Policies that parse threshold from string
# Use when: Policy has a single tunable numeric parameter (iterations, temperature, etc.).
# Naming convention: {policy}_{threshold} (e.g., "hgs_5000" → 5000 iterations)
# Parsed in: logic/src/policies/__init__.py:get_adapter()
THRESHOLD_POLICIES = [
    "vrpp",  # Parse MIP gap tolerance (e.g., "vrpp_0.01" → 1% optimality gap)
    "sans",  # Parse simulated annealing temperature (e.g., "sans_0.95" → cooling rate 0.95)
    "hgs",  # Parse max iterations (e.g., "hgs_10000" → 10k iterations)
    "alns",  # Parse max iterations (e.g., "alns_5000" → 5k destroy-repair cycles)
    "bcp",  # Parse time limit (e.g., "bcp_300" → 300 seconds)
]

# Policies with special config chars (e.g., lac_a_1.0, lac_b_2.0)
# Use when: Multiple named variants of same algorithm exist with different hyperparameters.
# Naming convention: {policy}_{char}_{threshold} (e.g., "lac_a_1.0" → LAC variant 'a' with threshold 1.0)
# Example use cases: Conservative vs aggressive variants, different operator sets, hybrid modes
CONFIG_CHAR_POLICIES = {
    "lac": ["a", "b"],  # Local Arc Consistency: 'a' (conservative SA), 'b' (aggressive SA)
}

# Simple name mappings (no threshold parsing)
# Maps tuple of display names → config file name (without policy_ prefix or .yaml).
# Use when: Policy has fixed configuration with no runtime tuning needed.
# Format: (alias1, alias2, ...) → config_name
SIMPLE_POLICIES = {
    # Neural network policies - all map to "neural" config
    ("am", "ddam", "transgcn"): "neural",  # Attention Model, Deep Decoder AM, Transformer GCN → policy_neural.yaml
    # Selection strategies - direct mapping
    ("last_minute",): "last_minute",  # Threshold-based collection → policy_last_minute.yaml
    ("regular",): "regular",  # Fixed-frequency collection → policy_regular.yaml
    # Classical solvers - direct mapping
    ("bcp",): "bcp",  # Branch-Cut-and-Price → policy_bcp.yaml
    ("lkh",): "lkh",  # Lin-Kernighan-Helsgaun → policy_lkh.yaml
    ("tsp",): "tsp",  # TSP solver (fast_tsp) → policy_tsp.yaml
    ("cvrp",): "cvrp",  # CVRP solver (OR-Tools) → policy_cvrp.yaml
}
