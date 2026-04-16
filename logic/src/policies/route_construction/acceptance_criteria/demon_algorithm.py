from typing import Any, Dict, List

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class DemonAlgorithm(IAcceptanceCriterion):
    """
    Demon Algorithm Acceptance Criterion (Wood, 2000).

    A deterministic alternative to Simulated Annealing that uses a "demon" credit budget.
    Improving moves populate the credit, and worsening moves are accepted only if
    authorized by the current credit balance.

    Mathematical Formulation (Minimization):
    - Improving Move (delta_f < 0): Always accepted; D = abs(delta_f).
    - Worsening Move (delta_f > 0): Accepted if delta_f <= D. If accepted, D = D - delta_f.
    """

    def __init__(self, warm_up_steps: int = 5, maximization: bool = False):
        """
        Args:
            warm_up_steps (int): Number of initial moves used to estimate initial demon credit.
            maximization (bool): Whether the problem is maximization.
        """
        self.warm_up_steps = warm_up_steps
        self.maximization = maximization

        self.demon_credit = 0.0
        self.history: List[float] = []
        self._warmed_up = False

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        # Always accept improving or equal moves (delta <= 0 in minimization)
        if delta <= 0:
            return True

        # If not warmed up, we typically reject worsening unless credit was somehow set
        if not self._warmed_up and self.demon_credit <= 0:
            return False

        # Worsening move: accepted if delta <= credit
        return delta <= self.demon_credit

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        # State update logic
        if not self._warmed_up:
            self.history.append(abs(delta))
            if len(self.history) >= self.warm_up_steps:
                self.demon_credit = max(self.history) if self.history else 0.0
                self._warmed_up = True
            return

        # Regular operation
        if delta < 0:
            # Improvement: The demon "recharges" with the improvement magnitude
            self.demon_credit = abs(delta)
        elif accepted and delta > 0:
            # Worsening move accepted: Credit is depleted by the cost
            self.demon_credit = max(0.0, self.demon_credit - delta)

    def get_state(self) -> Dict[str, Any]:
        return {
            "demon_credit": self.demon_credit,
            "warmed_up": self._warmed_up,
            "maximization": self.maximization,
        }
