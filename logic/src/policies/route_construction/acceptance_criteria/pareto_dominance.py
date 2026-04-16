"""
Pareto Dominance Acceptance Criterion.
"""

from typing import Any, Dict, Sequence

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("pd")
class ParetoDominanceAcceptance(IAcceptanceCriterion):
    """
    Accepts a candidate only if it strictly dominates the current solution across
    all objectives (Pareto superiority).

    This expects the `current_obj` and `candidate_obj` parameters to actually be
    iterables of floats (e.g., tuple or list of objective values) rather than scalar floats.
    """

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        # We type ignore because the base Interface dictates float,
        # but multi-objective needs tuples/lists.
        cur_objs: Sequence[float] = current_obj  # type: ignore
        cand_objs: Sequence[float] = candidate_obj  # type: ignore

        if len(cur_objs) != len(cand_objs):
            raise ValueError("Objective dimensions must match for Pareto dominance.")

        # Candidate dominates if it is >= in all objectives, and > in at least one
        # Assuming maximization for all objectives (as per VRPP structure)
        better_in_one = False
        for cur_val, cand_val in zip(cur_objs, cand_objs, strict=True):
            if cand_val < cur_val:
                return False
            if cand_val > cur_val:
                better_in_one = True

        return better_in_one

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
