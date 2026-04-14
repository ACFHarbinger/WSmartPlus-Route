r"""
Analytical recourse subproblem evaluator for the Integer L-Shaped Method.

The Stage 2 recourse function Q(ŷ, ω) is separable per bin and linear in the
binary visit decisions ŷ.  This permits an exact closed-form evaluation without
any LP solver, reducing the entire subproblem to a vectorized NumPy computation.

Mathematical Foundation (Laporte & Louveaux, 1993):

    Q(ŷ, ω) = Σᵢ [ p⁺ᵢ(ω) · (1 − ŷᵢ)  +  p⁻ᵢ(ω) · ŷᵢ ]

    where
        p⁺ᵢ(ω) = overflow_penalty  · max(0, wᵢ(ω) − τ)
        p⁻ᵢ(ω) = undervisit_penalty · max(0, τ − wᵢ(ω))
        τ       = collection_threshold  (fill % trigger level)
        wᵢ(ω)  = realised absolute fill level for bin i under scenario ω

Expected recourse over S uniform scenarios:
    Q̄(ŷ) = (1/S) Σ_ω Q(ŷ, ω)
           = Σᵢ [ Ēᵢ⁺ · (1 − ŷᵢ)  +  Ēᵢ⁻ · ŷᵢ ]

    where  Ēᵢ⁺ = (1/S) Σ_ω p⁺ᵢ(ω)   and   Ēᵢ⁻ = (1/S) Σ_ω p⁻ᵢ(ω).

Benders optimality cut (L-shaped cut):
    dᵢ  = ∂Q̄/∂ŷᵢ = Ēᵢ⁻ − Ēᵢ⁺   (gradient of expected recourse w.r.t. visit)
    e   = Q̄(ŷ̂) − Σᵢ dᵢ · ŷᵢ    (constant term)
    ⟹  θ ≥ e + Σᵢ dᵢ yᵢ          (cut added to master problem)

References:
    Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
    stochastic integer programs with complete recourse". Operations Research
    Letters, 13(3), 133-142.
"""

from typing import Dict, List, Tuple


class RecourseEvaluator:
    r"""Analytical evaluator for the Stage 2 recourse function Q̄(ŷ).

    Since the subproblem is separable per bin and linear in the binary visit
    decisions ŷ, the expected recourse and its gradient are computed directly
    from the scenario fill levels — no LP solver is required.

    The evaluator generates the coefficients (e, d) for the Benders optimality
    cut θ ≥ e + Σᵢ dᵢ yᵢ that is added to the master problem whenever the
    surrogate underestimates the expected recourse: Q̄(ŷ̂) > θ̂ + benders_gap.

    References:
        Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
        stochastic integer programs with complete recourse". Operations Research
        Letters, 13(3), 133-142.
    """

    def evaluate(
        self,
        y_hat: Dict[int, float],
        scenarios: List[Dict[int, float]],
        overflow_penalty: float,
        undervisit_penalty: float,
        collection_threshold: float,
    ) -> Tuple[float, float, Dict[int, float]]:
        r"""Compute Q̄(ŷ) and Benders optimality cut coefficients.

        Args:
            y_hat: Current master problem visit decisions {node_idx: value}.
                   Values are integers (0 or 1) extracted from the Gurobi solution.
            scenarios: SAA scenario list of length S.
                       Each entry is {node_idx: fill_%} for one realisation ω.
            overflow_penalty: Cost p⁺ per %-fill unit of unvisited bin overflow
                               above the collection threshold τ.
            undervisit_penalty: Cost p⁻ per %-fill unit below τ for visited bins.
            collection_threshold: Collection trigger τ (percent fill level).

        Returns:
            Tuple of (Q_bar, e, d):
                Q_bar: Expected recourse cost E_ω[Q(ŷ, ω)] at the current solution.
                e: Constant term of the Benders cut (θ ≥ e + Σᵢ dᵢ yᵢ).
                d: Dict {node_idx: dᵢ} of linear coefficients for the Benders cut.
        """
        n_scenarios = len(scenarios)
        if n_scenarios == 0:
            return 0.0, 0.0, {}

        node_ids = list(y_hat.keys())

        # ------------------------------------------------------------------
        # Step 1: Compute per-node expected overflow (Ēᵢ⁺) and undervisit
        #         (Ēᵢ⁻) costs averaged over S scenarios.
        # ------------------------------------------------------------------
        E_overflow: Dict[int, float] = {i: 0.0 for i in node_ids}
        E_undervisit: Dict[int, float] = {i: 0.0 for i in node_ids}

        for scenario in scenarios:
            for node_idx in node_ids:
                w = scenario.get(node_idx, 0.0)
                # Overflow penalty: bin NOT visited and fill exceeds threshold
                E_overflow[node_idx] += overflow_penalty * max(0.0, w - collection_threshold)
                # Undervisit penalty: bin IS visited but fill is below threshold
                E_undervisit[node_idx] += undervisit_penalty * max(0.0, collection_threshold - w)

        inv_s = 1.0 / n_scenarios
        for node_idx in node_ids:
            E_overflow[node_idx] *= inv_s
            E_undervisit[node_idx] *= inv_s

        # ------------------------------------------------------------------
        # Step 2: Evaluate Q̄(ŷ̂) at the current master problem solution.
        #         Q̄(ŷ) = Σᵢ [ Ēᵢ⁺·(1−ŷᵢ) + Ēᵢ⁻·ŷᵢ ]
        # ------------------------------------------------------------------
        Q_bar = 0.0
        for node_idx in node_ids:
            y_i = float(y_hat.get(node_idx, 0.0))
            Q_bar += E_overflow[node_idx] * (1.0 - y_i)
            Q_bar += E_undervisit[node_idx] * y_i

        # ------------------------------------------------------------------
        # Step 3: Compute Benders cut coefficients.
        #         dᵢ = ∂Q̄/∂ŷᵢ = Ēᵢ⁻ − Ēᵢ⁺
        #         e  = Q̄(ŷ̂) − Σᵢ dᵢ · ŷᵢ
        # ------------------------------------------------------------------
        d: Dict[int, float] = {node_idx: E_undervisit[node_idx] - E_overflow[node_idx] for node_idx in node_ids}

        e = Q_bar - sum(d[i] * float(y_hat.get(i, 0.0)) for i in node_ids)

        return Q_bar, e, d
