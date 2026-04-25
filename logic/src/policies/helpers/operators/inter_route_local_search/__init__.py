"""
Inter-route Local Search Operators.

This package contains operators that perform moves between different routes.

Attributes:
    move_swap_star: SWAP* inter-route node exchange.
    move_2opt_star: 2-opt* route tail exchange.
    move_3opt_star: 3-opt* route tail exchange.
    move_kopt_star: General k-opt* route tail exchange.
    cross_exchange: CROSS-exchange segment swap.
    improved_cross_exchange: CROSS-exchange with optional inversion.
    lambda_interchange: λ-interchange operator.
    ejection_chain: Ejection chain for infeasible nodes.
    cyclic_transfer: Cyclic node transfer across routes.
    exchange_2_0: Exchange (2,0) — move 2 nodes to another route.
    exchange_2_1: Exchange (2,1) — swap 2 nodes with 1 node.
    exchange_k_0: Exchange (k,0) — move k nodes to another route.
    exchange_k_h: Exchange (k,h) — swap k nodes with h nodes.
    shift_2_0: Shift(2,0) neighbourhood from Subramanian et al.
    swap_2_1: Swap(2,1) neighbourhood from Subramanian et al.
    swap_2_2: Swap(2,2) neighbourhood from Subramanian et al.
    move_cross: Cross neighbourhood (suffix exchange).

Example:
    >>> from logic.src.policies.helpers.operators.inter_route_local_search import (
    ...     move_2opt_star, cross_exchange, cyclic_transfer
    ... )
    >>> improved = move_2opt_star(ls, cuts=[(u, r_u, p_u), (v, r_v, p_v)])
"""

from .cross_exchange import (
    cross_exchange,
    improved_cross_exchange,
    lambda_interchange,
)
from .cyclic_transfer import cyclic_transfer
from .ejection_chain import ejection_chain
from .exchange_chain import (
    exchange_2_0,
    exchange_2_1,
    exchange_k_0,
    exchange_k_h,
)
from .k_opt_star import (
    move_2opt_star,
    move_3opt_star,
    move_kopt_star,
)
from .subramanian_neighborhoods import (
    move_cross,
    shift_2_0,
    swap_2_1,
    swap_2_2,
)
from .swap_star import move_swap_star

__all__ = [
    "move_swap_star",
    "move_2opt_star",
    "move_3opt_star",
    "move_kopt_star",
    "cross_exchange",
    "improved_cross_exchange",
    "lambda_interchange",
    "ejection_chain",
    "cyclic_transfer",
    "exchange_2_0",
    "exchange_2_1",
    "exchange_k_0",
    "exchange_k_h",
    "shift_2_0",
    "swap_2_1",
    "swap_2_2",
    "move_cross",
]
