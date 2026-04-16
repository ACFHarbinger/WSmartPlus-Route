from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.interfaces.distance_metric import IDistanceMetric


class SkewedVNSAcceptance(IAcceptanceCriterion):
    """
    Skewed Variable Neighborhood Search (SVNS) Acceptance Criterion.

    Accepts a candidate solution if it is either better than the current solution
    or if it is sufficiently distant (structurally diverse) to justify a slight
    objective deterioration. This prevents the search from being trapped in
    isomorphic basins.

    Mathematical Formulation (Maximization):
    Accept if: f_cand > f_cur - alpha * rho(x_cur, x_cand)
    where rho is the structural distance calculated by an IDistanceMetric.
    """

    def __init__(self, alpha: float, metric: IDistanceMetric, maximization: bool = True):
        """
        Args:
            alpha (float): Scaling factor for structural distance.
            metric (IDistanceMetric): Metric for computing structural distance rho.
            maximization (bool): Whether the problem is a maximization problem.
        """
        self.alpha = alpha
        self.metric = metric
        self.maximization = maximization

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        """
        Args:
            current_obj (float): Objective of the current solution.
            candidate_obj (float): Objective of the candidate solution.
            **kwargs: Must contain 'current_sol' and 'candidate_sol' for distance calculation.
        """
        if self.maximization:
            if candidate_obj > current_obj:
                return True
        else:
            if candidate_obj < current_obj:
                return True

        current_sol = kwargs.get("current_sol")
        candidate_sol = kwargs.get("candidate_sol")

        if current_sol is None or candidate_sol is None:
            # Fallback to only improving if structural data is missing
            return False

        rho = self.metric.compute(current_sol, candidate_sol)

        if self.maximization:
            return candidate_obj > (current_obj - self.alpha * rho)
        else:
            return candidate_obj < (current_obj + self.alpha * rho)

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"alpha": self.alpha}
