"""
HGS Local Search operators package.

This package provides a comprehensive suite of local search operators for VRP,
organized by category:
- Crossover: GA evolution recombination operators.
- Destroy: Removal heuristics (random, worst, cluster, shaw, string, route,
  neighbor, historical, sector).
- Repair: Insertion heuristics (greedy, regret-2, regret-k, blink, savings, deep).
- Intra-route: Moves within a single route (relocate, swap, 2-opt, 3-opt,
  k-opt, or-opt, geni, k-perm, link-swap, relocate-chain).
- Inter-route: Moves between different routes (swap*, k-opt*, cross, i-cross,
  ejection, lambda, cyclic, relocate (2,0)/(2,1)).
- Perturbation: Kick, perturb, double-bridge, genetic transformation, evolutionary.
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
    historical_removal,
    neighbor_removal,
    random_removal,
    route_removal,
    sector_removal,
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
    cyclic_transfer,
    ejection_chain,
    i_cross_exchange,
    lambda_interchange,
    move_2opt_star,
    move_3opt_star,
    move_kopt_star,
    move_swap_star,
    relocate_2_0,
    relocate_2_1,
    relocate_k_0,
    relocate_k_h,
)

# Intra-route operators
from .intra_route import (
    geni_insert,
    k_permutation,
    link_swap,
    move_2opt_intra,
    move_3opt_intra,
    move_kopt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
    relocate_chain,
    three_permutation,
)

# Perturbation operators
from .perturbation import (
    double_bridge,
    evolutionary_perturbation,
    genetic_transformation,
    kick,
    perturb,
)

# Repair operators
from .repair import (
    deep_insertion,
    greedy_insertion,
    greedy_insertion_with_blinks,
    regret_2_insertion,
    regret_k_insertion,
    savings_insertion,
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
    "route_removal",
    "neighbor_removal",
    "historical_removal",
    "sector_removal",
    # Repair
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
    "savings_insertion",
    "deep_insertion",
    # Intra-route
    "move_relocate",
    "move_swap",
    "move_2opt_intra",
    "move_3opt_intra",
    "move_kopt_intra",
    "move_or_opt",
    "geni_insert",
    "k_permutation",
    "link_swap",
    "relocate_chain",
    "three_permutation",
    # Inter-route
    "move_swap_star",
    "move_2opt_star",
    "move_3opt_star",
    "move_kopt_star",
    "cross_exchange",
    "i_cross_exchange",
    "ejection_chain",
    "lambda_interchange",
    "cyclic_transfer",
    "relocate_2_0",
    "relocate_2_1",
    "relocate_k_0",
    "relocate_k_h",
    # Perturbation
    "perturb",
    "kick",
    "double_bridge",
    "genetic_transformation",
    "evolutionary_perturbation",
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
