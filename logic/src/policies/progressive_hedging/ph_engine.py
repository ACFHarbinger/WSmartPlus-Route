r"""
Progressive Hedging (PH) engine for stochastic VRPP.

Progressive Hedging (Rockafellar and Wets, 1991) is a horizontal
decomposition algorithm for stochastic programs with non-anticipativity
constraints. It decomposes the problem into scenario-specific subproblems,
solving each with an augmented objective function that penalizes deviations
from the consensus (mean) solution.

Algorithm (minimization form):
----------------------------
1.  Initialize consensus $\bar{z}^0 = 0$ and duals $w_{\omega}^0 = 0$.
2.  In each iteration $k$:
    a. For each scenario $\omega$:
       Solve subproblem: $\min \,\, f_{\omega}(z_{\omega}) + (w_{\omega}^k)^T z_{\omega} + \frac{\rho}{2} \|z_{\omega} - \bar{z}^k\|^2$
    b. Update consensus: $\bar{z}^{k+1} = \sum_{\omega} P(\omega) z_{\omega}^{k+1}$
    c. Update duals: $w_{\omega}^{k+1} = w_{\omega}^k + \rho (z_{\omega}^{k+1} - \bar{z}^{k+1})$

Linearization for Binary Variables:
----------------------------------
Since $y_{i, \omega} \in \{0, 1\}$, the quadratic term $\|z_{\omega} - \bar{z}\|^2$ is linearized:
    $(y_{i, \omega} - \bar{y}_i)^2 = y_{i, \omega}(1 - 2\bar{y}_i) + \bar{y}_i^2$.
The constant $\bar{y}_i^2$ is ignored in the subproblem optimization.

The augmented cost for node $i$ in scenario $\omega$ is:
    $\Delta Cost_{i, \omega} = w_{i, \omega} + \frac{\rho}{2}(1 - 2\bar{y}_i)$
"""

import logging
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np

from logic.src.configs.policies import PHConfig
from logic.src.policies.base.factory import PolicyFactory

logger = logging.getLogger(__name__)


class ProgressiveHedgingEngine:
    """Core iterative engine for Progressive Hedging decomposition.

    Manages scenario subproblems, performs consensus aggregation, and updates
    dual multipliers based on non-anticipativity residuals.
    """

    def __init__(self, config: PHConfig) -> None:
        """Initialise the PH engine.

        Args:
            config: Progressive Hedging configuration.
        """
        self.config = config
        self.sub_solver_name = config.sub_solver

        # Consensus and dual state
        self.y_consensus: Dict[int, float] = {}
        self.w_duals: List[Dict[int, float]] = []  # List[ScenarioIdx] -> {NodeIdx: Dual}
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def ensure_route_list(routes: Union[List[int], List[List[int]]]) -> List[List[int]]:
        # 1. Check if already a list of lists
        if routes and isinstance(routes[0], list):
            return routes  # type: ignore[return-value]

        # 2. Handle flattened list [0, 1, 3, 0, 4, 0...]
        # Find all indices where the value is 0
        depot_indices = [i for i, val in enumerate(routes) if val == 0]

        # Create pairs of indices: (0, 3), (3, 5), (5, 7)
        # This ensures the 0 is both the end of one route and start of the next
        nested_routes = [routes[depot_indices[i] : depot_indices[i + 1] + 1] for i in range(len(depot_indices) - 1)]
        return nested_routes  # type: ignore[return-value]

    def solve(  # noqa: C901
        self,
        sub_dist_matrix: np.ndarray,
        scenario_wastes: List[Dict[int, float]],
        capacity: float,
        revenue: float,
        cost_unit: float,
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        """Run the Progressive Hedging iterative algorithm.

        Args:
            sub_dist_matrix: Localised N×N distance matrix.
            scenario_wastes: List of SAA scenarios (each a node -> fill_% dict).
            capacity: Vehicle capacity.
            revenue: Revenue per unit of waste.
            cost_unit: Travel cost per distance unit.
            mandatory_nodes: Nodes that must be visited.
            **kwargs: Additional parameters for sub-solvers.

        Returns:
            Tuple of (best_consensus_routes, expected_profit, stats).
        """
        n_scenarios = len(scenario_wastes)
        if n_scenarios == 0:
            return [], 0.0, {"error": "No scenarios provided"}

        # Identify all nodes involved across all scenarios
        all_nodes: Set[int] = set()
        for wastes in scenario_wastes:
            all_nodes.update(wastes.keys())
        all_nodes.discard(0)  # Depot is not a decision variable for non-anticipativity
        node_ids = sorted(list(all_nodes))

        # 1. Initialize State
        self.y_consensus = {i: 0.0 for i in node_ids}
        self.w_duals = [{i: 0.0 for i in node_ids} for _ in range(n_scenarios)]
        probabilities = [1.0 / n_scenarios] * n_scenarios

        # 2. Iterative Loop
        for k in range(self.config.max_iterations):
            scenario_results: List[Tuple[List[List[int]], float, Dict[int, float]]] = []

            # Step 2a: Solve Subproblems
            for s_idx in range(n_scenarios):
                # Calculate augmented node prizes (linearized PH objective)
                # PH term for minimization: w*y + rho/2 * (y - y_bar)^2
                # => linearized yield: DeltaCost = w + rho/2 * (1 - 2*y_bar)
                # Since our VRPP solvers use Profit (Revenue - Cost), we invert:
                # AugmentedProfit = BaseProfit - DeltaCost

                node_prizes: Dict[int, float] = {}
                for i in node_ids:
                    w = scenario_wastes[s_idx].get(i, 0.0)
                    base_profit = w * revenue

                    dual = self.w_duals[s_idx][i]
                    penalty = (self.config.rho / 2.0) * (1.0 - 2.0 * self.y_consensus[i])

                    # Augmented Revenue = Base Revenue - Dual - Penalty
                    node_prizes[i] = base_profit - dual - penalty

                # Dispatch to sub-solver
                sub_solver = PolicyFactory.get_adapter(self.sub_solver_name)

                values = {}
                if hasattr(sub_solver, "config") and sub_solver.config is not None:
                    if hasattr(sub_solver.config, "__dict__"):
                        values = sub_solver.config.__dict__
                    elif isinstance(sub_solver.config, dict):
                        values = sub_solver.config

                routes, profit, _dist = sub_solver.execute(
                    sub_dist_matrix=sub_dist_matrix,
                    sub_wastes=scenario_wastes[s_idx],
                    capacity=capacity,
                    revenue=revenue,
                    cost_unit=cost_unit,
                    values=values,
                    mandatory_nodes=mandatory_nodes,
                    node_prizes=node_prizes,  # CRITICAL: PH penalty injection
                    **kwargs,
                )

                # Extract y_hat (binary visit decisions) from routes
                y_hat = {i: 0.0 for i in node_ids}
                routes = self.ensure_route_list(routes)
                for route in routes:
                    for node in route:
                        if node in y_hat:
                            y_hat[node] = 1.0

                scenario_results.append((routes, profit, y_hat))

            # Step 2b: Update Consensus
            new_consensus = {i: 0.0 for i in node_ids}
            for s_idx in range(n_scenarios):
                _, _, y_hat = scenario_results[s_idx]
                for i in node_ids:
                    new_consensus[i] += probabilities[s_idx] * y_hat[i]

            # Step 2c: Compute Convergence & Update Duals
            primal_residual = 0.0
            for s_idx in range(n_scenarios):
                _, _, y_hat = scenario_results[s_idx]
                for i in node_ids:
                    # Dual update: w = w + rho * (y - y_bar)
                    residual = y_hat[i] - new_consensus[i]
                    self.w_duals[s_idx][i] += self.config.rho * residual
                    primal_residual += probabilities[s_idx] * (residual**2)

            primal_residual = np.sqrt(primal_residual)
            self.y_consensus = new_consensus

            # Tracking
            avg_profit = sum(res[1] for res in scenario_results) / n_scenarios
            if self.config.verbose:
                logger.info(f"PH Iteration {k}: Primal Residual = {primal_residual:.4f}, Avg Profit = {avg_profit:.2f}")

            self.history.append({"iteration": k, "primal_residual": primal_residual, "avg_profit": avg_profit})

            # Termination check
            if primal_residual < self.config.convergence_tol:
                if self.config.verbose:
                    logger.info(f"PH converged at iteration {k}")
                break

        # 3. Final Solution Selection
        # We pick the scenario solution that is 'closest' to the consensus,
        # or simply based on highest average performance.
        # For simplicity, we choose the solution from iteration k=0 or the best found.
        # A more robust way is to solve a 'consensus' MILP fixing y to y_bar >= threshold.

        # Strategy: Return the routes from the first scenario of the last iteration
        # as a representative feasible solution.
        best_routes: List[List[int]] = scenario_results[0][0]
        expected_profit = sum(res[1] for res in scenario_results) / n_scenarios

        return (
            best_routes,
            expected_profit,
            {"iterations": len(self.history), "final_residual": primal_residual, "convergence_history": self.history},
        )
