from abc import ABC, abstractmethod
from typing import Any, Dict


class IAcceptanceCriterion(ABC):
    """
    Abstract Base Class for modular move acceptance strategies.
    """

    @abstractmethod
    def setup(self, initial_objective: float) -> None:
        """
        Initializes the internal state of the criterion based on the starting solution.

        Args:
            initial_objective (float): The objective value of the starting solution.
        """
        pass

    @abstractmethod
    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        """
        Evaluates whether a candidate solution should be accepted.
        This method MUST be stateless and purely functional.

        Args:
            current_obj (float): The objective value of the current incumbent solution.
            candidate_obj (float): The objective value of the proposed candidate solution.
            **kwargs (Any): Additional context such as structural solution representations.

        Returns:
            bool: True if the candidate is accepted, False otherwise.
        """
        pass

    @abstractmethod
    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        """
        Advances the internal state of the criterion (e.g., cooling temperature,
        updating memory buffers, adjusting thresholds).

        Args:
            current_obj (float): The objective value of the solution *after* the decision.
            candidate_obj (float): The objective value of the evaluated candidate.
            accepted (bool): The boolean result of the `accept` method.
            **kwargs (Any): Additional context such as structural solution representations.
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the internal variables for telemetry and experiment tracking.

        Returns:
            Dict[str, Any]: Dictionary of internal state variables.
        """
        return {}
