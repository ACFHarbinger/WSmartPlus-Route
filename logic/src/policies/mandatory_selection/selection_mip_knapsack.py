"""
MIP Multiple-Knapsack Selection Strategy Module.

Exact 0/1 multiple-knapsack formulation of mandatory bin selection solved
with ``scipy.optimize.milp``. Each knapsack is a vehicle whose capacity
is ``context.vehicle_capacity`` (waste mass). The objective is net
profit: revenue minus distance cost at rate ``context.cost_per_km``.

Formulation
-----------
    maximize    sum_{i,k} ( r_i - lambda * w_i^{dist} ) * x_{i,k}
    subject to  sum_k   x_{i,k}           <= 1      for all bins i
                sum_i   m_i * x_{i,k}     <= Q      for all vehicles k
                sum_i   x_{i,k} - sum_i x_{i,k+1}  >= 0   (symmetry break)
                x_{i,k} in {0, 1}

Because the marginal insertion cost does not linearize cleanly (it
depends on what else is packed into the same knapsack), this MIP uses
the static depot round-trip ``w_i^{dist} = 2 * d(depot, i)`` as the
distance proxy. This is conservative: it over-estimates insertion cost
for bins that would realistically be picked up along existing corridors,
so the MIP may under-select relative to a true VRP-aware optimum.

Symmetry breaking
-----------------
All K vehicles are identical, which induces a K! symmetry in the
branch-and-bound tree. We add lexicographic count constraints
``sum_i x_{i,k} >= sum_i x_{i,k+1}`` that force vehicle k to carry at
least as many bins as vehicle k+1, pruning most symmetric solutions.

Unbounded case
--------------
If ``context.n_vehicles <= 0`` the strategy short-circuits to all bins
with non-negative standalone net profit.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_mip_knapsack \\
    ...     import MIPKnapsackSelection
    >>> strategy = MIPKnapsackSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from logic.src.interfaces.mandatory import IMandatorySelectionStrategy
from logic.src.policies.helpers.mandatory.base.selection_context import SelectionContext
from logic.src.policies.helpers.mandatory.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("mip_knapsack")
class MIPKnapsackSelection(IMandatorySelectionStrategy):
    """Exact 0/1 multiple-knapsack selection via mixed-integer programming."""

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Solve the 0/1 multiple-knapsack problem and return the selected bins.

        Reads ``vehicle_capacity``, ``n_vehicles``, and (optionally)
        ``cost_per_km`` from the context.
        """
        if context.distance_matrix is None:
            raise ValueError("MIPKnapsackSelection requires a distance_matrix.")
        if context.revenue_kg <= 0:
            return []

        dm = np.asarray(context.distance_matrix)

        # Per-bin mass, revenue, and static distance proxy.
        bin_cap = context.bin_volume * context.bin_density
        mass_all = (context.current_fill / context.max_fill) * bin_cap
        revenue_all = mass_all * context.revenue_kg
        round_trip_all = 2.0 * np.asarray(dm[0, 1:], dtype=float)

        cost_per_km = float(getattr(context, "cost_per_km", 0.0))
        net_profit_all = revenue_all - cost_per_km * round_trip_all

        # Eligible = positive mass and positive net profit.
        eligible_idx = np.nonzero((mass_all > 0) & (net_profit_all > 0))[0]
        if eligible_idx.size == 0:
            return []

        n_vehicles = int(getattr(context, "n_vehicles", 1))

        # Unbounded knapsacks: no binding constraint, take everything profitable.
        if n_vehicles <= 0:
            return sorted((eligible_idx + 1).tolist())

        capacity = float(context.vehicle_capacity)
        if capacity <= 0:
            return []

        # Dominance prune: drop bins that do not fit any single vehicle.
        fits_any = mass_all[eligible_idx] <= capacity
        eligible_idx = eligible_idx[fits_any]
        if eligible_idx.size == 0:
            return []

        n = int(eligible_idx.size)
        K = n_vehicles
        p = net_profit_all[eligible_idx]
        m = mass_all[eligible_idx]

        # Decision variables flattened row-major: idx(i, k) = i * K + k.
        n_vars = n * K

        # Objective: minimize -sum_{i,k} p_i x_{i,k}  (==  maximize net profit).
        c_obj = -np.repeat(p, K)

        # Assignment: each bin is collected by at most one vehicle.
        A_assign = np.kron(np.eye(n), np.ones((1, K)))

        # Capacity: sum of masses in knapsack k is at most Q.
        A_cap = np.kron(m[None, :], np.eye(K))

        constraints = [
            LinearConstraint(A_assign, -np.inf, 1.0),
            LinearConstraint(A_cap, -np.inf, capacity),
        ]

        # Lexicographic symmetry break: count(k) >= count(k+1) for k = 0..K-2.
        # Row k encodes (sum_i x_{i,k}) - (sum_i x_{i,k+1}) >= 0.
        if K >= 2:
            A_sym = np.zeros((K - 1, n_vars))
            for k in range(K - 1):
                for i in range(n):
                    A_sym[k, i * K + k] = 1.0
                    A_sym[k, i * K + k + 1] = -1.0
            constraints.append(LinearConstraint(A_sym, 0.0, np.inf))

        integrality = np.ones(n_vars, dtype=int)  # all binary
        bounds = Bounds(lb=0, ub=1)

        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )

        if not result.success or result.x is None:
            return []

        x = np.asarray(result.x).reshape(n, K)
        taken_local = np.nonzero(x.sum(axis=1) > 0.5)[0]
        taken_global = eligible_idx[taken_local]
        return sorted((taken_global + 1).tolist())
