r"""
Integer L-Shaped (Benders Decomposition) Engine for Stochastic VRPP / SCWCVRP.

Implements the outer Benders decomposition loop that coordinates the Master
Problem (Gurobi MILP) and the analytical recourse subproblem evaluator.

Algorithm — Integer L-Shaped Method (Laporte & Louveaux, 1993):
    0. Generate S SAA scenarios {ω₁, …, ωₛ} of end-of-day bin fill levels.
    1. Build master problem with surrogate θ ≥ θ_LB.
    2. Repeat until convergence or iteration / time budget exhausted:
        a. Solve MP (with SEC lazy callbacks inside Gurobi) →
           integer routing (x̂, ŷ) and surrogate value θ̂.
        b. Evaluate Q̄(ŷ̂) analytically over SAA scenarios (no LP needed).
        c. If Q̄(ŷ̂) ≤ θ̂ + benders_gap: CONVERGED — break.
        d. Otherwise: generate optimality cut θ ≥ e + Σᵢ dᵢ yᵢ,
           add to MP, and re-solve.
    3. Return the best routing solution and its deterministic net profit.

Correctness Guarantees:
    - The subproblem is always feasible (affine in slack variables), so only
      optimality cuts are needed — no feasibility cuts.
    - Q̄(ŷ) is convex and piecewise-linear in ŷ; the gradient (d) is an exact
      supergradient, ensuring non-decreasing lower bounds across iterations.
    - Because Stage 1 variables are binary, the L-shaped cuts are *combinatorial*
      (valid at every integer point, not just at the LP relaxation).
    - The algorithm terminates finitely under a bound on the number of distinct
      binary ŷ solutions (exponential in |N| but bounded in practice by time/iter).

References:
    Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
    stochastic integer programs with complete recourse". Operations Research
    Letters, 13(3), 133-142.
"""

import time
from typing import Any, Dict, List, Tuple

from logic.src.policies.helpers.branching_solvers.vrpp_model import VRPPModel

from .master_problem import MasterProblem
from .params import ILSBDParams
from .scenario import ScenarioGenerator
from .subproblem import RecourseEvaluator

try:
    import gurobipy  # noqa: F401

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class IntegerLShapedEngine:
    r"""Benders decomposition engine for stochastic VRPP / SCWCVRP.

    Orchestrates the full Integer L-Shaped method, including:
        - SAA scenario generation   (ScenarioGenerator)
        - Iterative master-problem solves  (MasterProblem / Gurobi)
        - Analytical recourse evaluation and cut generation (RecourseEvaluator)
        - Convergence monitoring and time-budget tracking

    Usage example::

        from logic.src.policies.helpers.branching_solvers.vrpp_model import VRPPModel
        from logic.src.policies.integer_l_shaped.ils_engine import IntegerLShapedEngine
        from logic.src.policies.integer_l_shaped.params import ILSBDParams

        model = VRPPModel(n_nodes=..., cost_matrix=..., wastes=..., ...)
        engine = IntegerLShapedEngine(model, ILSBDParams())
        routes, y_hat, profit, stats = engine.solve(sub_wastes)

    Attributes:
        model: VRPPModel encoding the deterministic problem structure.
        params: ILSBDParams configuring all solver hyper-parameters.
        stats: Diagnostic dict populated during ``solve()``.
    """

    def __init__(self, model: VRPPModel, params: ILSBDParams) -> None:
        """Initialise the engine.

        Args:
            model: VRPPModel instance (network structure, costs, fill levels).
            params: ILSBDParams configuring Benders iterations, scenarios, penalties.
        """
        self.model = model
        self.params = params
        self.stats: Dict[str, Any] = {
            "benders_iterations": 0,
            "benders_cuts_added": 0,
            "converged": False,
            "total_time": 0.0,
            "final_Q_bar": None,
            "final_theta": None,
            "gurobi_available": GUROBI_AVAILABLE,
        }

        self._scenario_gen = ScenarioGenerator()
        self._evaluator = RecourseEvaluator()

    def solve(
        self,
        sub_wastes: Dict[int, float],
    ) -> Tuple[List[List[int]], Dict[int, float], float, Dict[str, Any]]:
        r"""Run the Integer L-Shaped Benders decomposition.

        Args:
            sub_wastes: Observed fill levels {local_node_idx: fill_%}.
                        Used as the Gamma distribution mean for scenario generation.

        Returns:
            Tuple of (routes, y_hat, profit, stats):
                routes: Customer-only route lists from the optimal MP solution.
                y_hat:  Dict {node_idx: 0.0/1.0} binary visit decisions.
                profit: Deterministic net profit = revenue − travel_cost.
                        The recourse surrogate θ is intentionally excluded so
                        that the returned profit matches the deterministic-policy
                        convention expected by BaseRoutingPolicy.
                stats:  Solver diagnostics (iterations, cuts, convergence, time).
        """
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi (gurobipy) is required for the Integer L-Shaped solver. "
                "Ensure gurobipy is installed and a valid licence is available."
            )

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 0: SAA scenario generation
        # ------------------------------------------------------------------
        seed = self.params.seed if self.params.seed is not None else 42
        scenarios = self._scenario_gen.generate(
            sub_wastes=sub_wastes,
            n_scenarios=self.params.n_scenarios,
            fill_rate_cv=self.params.fill_rate_cv,
            seed=seed,
        )

        # ------------------------------------------------------------------
        # Step 1: Build master problem (once)
        # ------------------------------------------------------------------
        master = MasterProblem(model=self.model, params=self.params)
        master.build()

        # Tracking state
        best_routes: List[List[int]] = []
        best_y_hat: Dict[int, float] = {}
        best_profit: float = -float("inf")
        self.stats["benders_iterations"] = 0
        self.stats["benders_cuts_added"] = 0
        self.stats["converged"] = False

        # ------------------------------------------------------------------
        # Step 2: Benders outer loop
        # ------------------------------------------------------------------
        for iteration in range(self.params.max_benders_iterations):
            self.stats["benders_iterations"] = iteration + 1

            # Respect wall-clock budget (keep 5 s reserve for extraction)
            elapsed = time.perf_counter() - t_start
            remaining = self.params.time_limit - elapsed
            if remaining < 5.0:
                break

            # Update Gurobi time limit to remaining wall-clock budget
            master._gurobi_model.Params.TimeLimit = remaining  # type: ignore[union-attr]

            # ---- 2a. Solve master problem --------------------------------
            routes, y_hat, theta_hat, obj_MP = master.solve()

            # No feasible solution at all: terminate early
            if not routes and all(v == 0.0 for v in y_hat.values()):
                break

            # Track best deterministic solution (excludes θ from profit)
            det_profit = self._compute_deterministic_profit(routes, y_hat)
            if det_profit > best_profit:
                best_profit = det_profit
                best_routes = [list(r) for r in routes]
                best_y_hat = dict(y_hat)

            # ---- 2b. Evaluate recourse ----------------------------------
            Q_bar, e, d = self._evaluator.evaluate(
                y_hat=y_hat,
                scenarios=scenarios,
                overflow_penalty=self.params.overflow_penalty,
                undervisit_penalty=self.params.undervisit_penalty,
                collection_threshold=self.params.collection_threshold,
            )

            self.stats["final_Q_bar"] = Q_bar
            self.stats["final_theta"] = theta_hat

            if self.params.verbose:
                print(
                    f"[ILS] iter={iteration + 1:3d}  "
                    f"θ̂={theta_hat:10.4f}  Q̄={Q_bar:10.4f}  "
                    f"gap={Q_bar - theta_hat:+.4f}  "
                    f"cuts={self.stats['benders_cuts_added']}"
                )

            # ---- 2c. Convergence check ----------------------------------
            if Q_bar <= theta_hat + self.params.benders_gap:
                self.stats["converged"] = True
                break

            # ---- 2d. Add Benders optimality cut -------------------------
            master.add_optimality_cut(e=e, d=d)
            self.stats["benders_cuts_added"] += 1

        # ------------------------------------------------------------------
        # Step 3: Finalise statistics
        # ------------------------------------------------------------------
        self.stats["total_time"] = time.perf_counter() - t_start
        # Sync cut count from master's authoritative counter
        self.stats["benders_cuts_added"] = int(master.stats["benders_cuts"])

        return best_routes, best_y_hat, best_profit, self.stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_deterministic_profit(
        self,
        routes: List[List[int]],
        y_hat: Dict[int, float],
    ) -> float:
        r"""Compute the deterministic net profit Σᵢ rᵢ ŷᵢ − Σᵢⱼ cᵢⱼ x̂ᵢⱼ.

        The recourse surrogate θ is intentionally excluded so the returned
        value is compatible with the deterministic-policy contract used by
        BaseRoutingPolicy (``profit = revenue − travel_cost``).

        Args:
            routes: Customer-only route lists from the master problem.
            y_hat:  Binary visit decisions {node_idx: 0.0/1.0}.

        Returns:
            Net profit in the problem's monetary units.
        """
        revenue = sum(self.model.get_node_profit(i) * float(y_hat.get(i, 0.0)) for i in self.model.customers)

        travel_cost = 0.0
        for route in routes:
            if not route:
                continue
            path = [self.model.depot] + route + [self.model.depot]
            for k in range(len(path) - 1):
                travel_cost += self.model.get_edge_cost(path[k], path[k + 1])

        return revenue - travel_cost
