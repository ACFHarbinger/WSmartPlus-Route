"""Great Deluge (GD) Acceptance Criterion.

Deterministic acceptance algorithm based on a rising 'water level' threshold.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("gd")
class GreatDelugeAcceptance(IAcceptanceCriterion):
    """Deterministic Great Deluge algorithm.

    Accepts a candidate if its objective is greater than or equal to a monotonically
    rising "water level". The water level rises linearly from the initial objective
    to a target objective over a predefined number of iterations.

    Attributes:
        target_multiplier (float): Expected relative improvement target.
        max_iterations (int): Total budget of iterations to reach the target.
        water_level (float): Current acceptance threshold.
        rain_speed (float): Increment added to the water level at each step.
    """

    def __init__(self, target_fitness_multiplier: float, max_iterations: int):
        """Initialize the Great Deluge criterion.

        Args:
            target_fitness_multiplier (float): The expected relative improvement
                (e.g., 1.1 implies a 10% expected improvement over the initial solution).
            max_iterations (int): The total budget of iterations to reach the target.
        """
        self.target_multiplier = target_fitness_multiplier
        self.max_iterations = max_iterations
        self.water_level = 0.0
        self.rain_speed = 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective (ObjectiveValue): The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.water_level = initial_objective
        target_objective = initial_objective * self.target_multiplier

        # Calculate exactly how much the water should rise per step
        if self.max_iterations > 0:
            self.rain_speed = (target_objective - initial_objective) / self.max_iterations
        else:
            self.rain_speed = 0.0

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept based on the current water level.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            **kwargs (Any): Additional context (not used).

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate improves the water level.
                - metrics (AcceptanceMetrics): State metadata including water level.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj >= self.water_level)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "water_level": self.water_level,
            "rain_speed": self.rain_speed,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Increment the water level by the rain speed.

        Args:
            current_obj (ObjectiveValue): Objective of the previous solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            accepted (bool): Whether the candidate was accepted.
            **kwargs (Any): Additional context (not used).
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.water_level += self.rain_speed

    def get_state(self) -> Dict[str, Any]:
        """Return the current water level and rain speed.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"water_level": self.water_level, "rain_speed": self.rain_speed}
