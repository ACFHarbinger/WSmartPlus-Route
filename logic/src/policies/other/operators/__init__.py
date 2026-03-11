"""
HGS Local Search operators package.

This package provides a comprehensive suite of local search operators for VRP,
organized by category:
- Crossover: GA evolution recombination operators.
- Destroy: Removal heuristics (random, worst, cluster, shaw, string).
- Repair: Insertion heuristics (greedy, regret-2, regret-k, blink).
- Intra-route: Moves within a single route (relocate, swap, 2-opt, 3-opt, or-opt).
- Inter-route: Moves between different routes (swap*, 2-opt*, cross, ejection, lambda).
- Perturbation: Kick, random perturb, and unstringing operators.
- Heuristics: Complex local search heuristics (initialization, LKH).
"""

# Crossover operators
from .crossover import (
    CROSSOVER_NAMES,
    CROSSOVER_OPERATORS,
    edge_recombination_crossover,
    generalized_partition_crossover,
    ordered_crossover,
    position_independent_crossover,
    selective_route_exchange_crossover,
)

# Destroy operators
from .destroy import (
    cluster_removal,
    random_removal,
    shaw_removal,
    string_removal,
    worst_removal,
)

# Heuristics
from .heuristics import (
    build_greedy_routes,
    build_nn_routes,
    solve_lkh,
)

# Inter-route operators
from .inter_route import (
    cross_exchange,
    ejection_chain,
    lambda_interchange,
    move_2opt_star,
    move_swap_star,
)

# Intra-route operators
from .intra_route import (
    move_2opt_intra,
    move_3opt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
)

# Perturbation operators
from .perturbation import (
    kick,
    perturb,
)

# Repair operators
from .repair import (
    greedy_insertion,
    greedy_insertion_with_blinks,
    regret_2_insertion,
    regret_k_insertion,
)

# Unstringing and stringing (US)
from .unstringing_stringing import (
    apply_type_i_s,
    apply_type_i_us,
    apply_type_ii_s,
    apply_type_ii_us,
    apply_type_iii_s,
    apply_type_iii_us,
    apply_type_iv_s,
    apply_type_iv_us,
)

__all__ = [
    # Crossover
    "CROSSOVER_NAMES",
    "CROSSOVER_OPERATORS",
    "edge_recombination_crossover",
    "generalized_partition_crossover",
    "ordered_crossover",
    "position_independent_crossover",
    "selective_route_exchange_crossover",
    # Destroy
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
    # Repair
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
    # Intra-route
    "move_relocate",
    "move_swap",
    "move_2opt_intra",
    "move_3opt_intra",
    "move_or_opt",
    # Inter-route
    "move_swap_star",
    "move_2opt_star",
    "cross_exchange",
    "ejection_chain",
    "lambda_interchange",
    # Perturbation
    "perturb",
    "kick",
    # Unstringing and stringing (US)
    "apply_type_i_us",
    "apply_type_ii_us",
    "apply_type_iii_us",
    "apply_type_iv_us",
    "apply_type_i_s",
    "apply_type_ii_s",
    "apply_type_iii_s",
    "apply_type_iv_s",
    # Heuristics
    "build_greedy_routes",
    "build_nn_routes",
    "solve_lkh",
]
