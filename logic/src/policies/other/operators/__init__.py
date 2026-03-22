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
    historical_profit_removal,
    historical_removal,
    neighbor_removal,
    random_removal,
    route_profit_removal,
    route_removal,
    sector_removal,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
    worst_profit_removal,
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
    exchange_2_0,
    exchange_2_1,
    exchange_k_0,
    exchange_k_h,
    improved_cross_exchange,
    lambda_interchange,
    move_2opt_star,
    move_3opt_star,
    move_kopt_star,
    move_swap_star,
)

# Intra-route operators
from .intra_route import (
    k_permutation,
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
    evolutionary_perturbation_profit,
    genetic_transformation,
    genetic_transformation_profit,
    kick,
    kick_profit,
    perturb,
)

# Repair operators
from .repair import (
    deep_insertion,
    deep_profit_insertion,
    farthest_insertion,
    geni_insertion,
    greedy_insertion,
    greedy_insertion_with_blinks,
    greedy_profit_insertion,
    greedy_profit_insertion_with_blinks,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_k_insertion,
    regret_k_profit_insertion,
    savings_insertion,
    savings_profit_insertion,
)

# Unstringing and stringing (US)
from .unstringing_stringing import (
    apply_type_i_s,
    apply_type_i_s_profit,
    apply_type_i_us,
    apply_type_i_us_profit,
    apply_type_ii_s,
    apply_type_ii_s_profit,
    apply_type_ii_us,
    apply_type_ii_us_profit,
    apply_type_iii_s,
    apply_type_iii_s_profit,
    apply_type_iii_us,
    apply_type_iii_us_profit,
    apply_type_iv_s,
    apply_type_iv_s_profit,
    apply_type_iv_us,
    apply_type_iv_us_profit,
    stringing_insertion,
    stringing_profit_insertion,
    unstringing_profit_removal,
    unstringing_removal,
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
    "worst_profit_removal",
    "shaw_profit_removal",
    "route_profit_removal",
    "neighbor_profit_removal",
    "historical_profit_removal",
    "sector_profit_removal",
    # Repair
    "greedy_insertion",
    "greedy_profit_insertion",
    "regret_2_insertion",
    "regret_2_profit_insertion",
    "regret_k_insertion",
    "regret_k_profit_insertion",
    "greedy_insertion_with_blinks",
    "greedy_profit_insertion_with_blinks",
    "savings_insertion",
    "savings_profit_insertion",
    "deep_insertion",
    "deep_profit_insertion",
    "farthest_insertion",
    "geni_insertion",
    # Intra-route
    "move_relocate",
    "move_swap",
    "move_2opt_intra",
    "move_3opt_intra",
    "move_kopt_intra",
    "move_or_opt",
    "k_permutation",
    "relocate_chain",
    "three_permutation",
    # Inter-route
    "move_swap_star",
    "move_2opt_star",
    "move_3opt_star",
    "move_kopt_star",
    "cross_exchange",
    "improved_cross_exchange",
    "ejection_chain",
    "lambda_interchange",
    "cyclic_transfer",
    "exchange_2_0",
    "exchange_2_1",
    "exchange_k_0",
    "exchange_k_h",
    # Perturbation
    "perturb",
    "kick",
    "kick_profit",
    "double_bridge",
    "genetic_transformation",
    "genetic_transformation_profit",
    "evolutionary_perturbation",
    "evolutionary_perturbation_profit",
    # Unstringing and stringing (US)
    "apply_type_i_us",
    "apply_type_i_us_profit",
    "apply_type_ii_us",
    "apply_type_ii_us_profit",
    "apply_type_iii_us",
    "apply_type_iii_us_profit",
    "apply_type_iv_us",
    "apply_type_iv_us_profit",
    "apply_type_i_s",
    "apply_type_i_s_profit",
    "apply_type_ii_s",
    "apply_type_ii_s_profit",
    "apply_type_iii_s",
    "apply_type_iii_s_profit",
    "apply_type_iv_s",
    "apply_type_iv_s_profit",
    "stringing_insertion",
    "stringing_profit_insertion",
    "unstringing_removal",
    "unstringing_profit_removal",
    # Heuristics
    "build_greedy_routes",
    "build_nn_routes",
    "solve_lkh",
]
