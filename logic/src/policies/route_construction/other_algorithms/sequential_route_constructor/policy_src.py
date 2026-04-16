"""
SRC (Sequential Route Constructor) — Meta-Constructor for Routing.
"""

import logging
import time
from typing import Any, List, Optional, Tuple, Union

from logic.src.interfaces.route_constructor import IRouteConstructor
from logic.src.policies.context.search_context import SearchContext

from ...base.base_routing_policy import BaseRoutingPolicy
from ...base.registry import RouteConstructorRegistry
from .params import SRCParams

logger = logging.getLogger(__name__)


@RouteConstructorRegistry.register("src")
class SequentialRouteConstructor(BaseRoutingPolicy):
    """
    Sequential Route Constructor (SRC) that runs multiple routing policies in sequence.

    Each subsequent policy uses the output of the previous one as a starting
    point (passed via ``kwargs['tour']``, ``kwargs['cost']``, and ``kwargs['search_context']``).
    """

    def __init__(self, config: Any = None):
        """
        Initialize the Sequential Route Constructor.

        Args:
            config: Either a SRCConfig dataclass or a dict.
        """
        super().__init__(config)
        self.constructors: List[IRouteConstructor] = []
        self._initialized = False

        # Load params
        if self.config is not None:
            self.params = SRCParams.from_config(self.config)
        else:
            self.params = SRCParams(constructors=["tsp", "nn"])

    @classmethod
    def _config_class(cls):
        from logic.src.configs.policies.src import SRCConfig

        return SRCConfig

    @classmethod
    def _get_config_key(cls) -> str:
        return "src"

    def _initialize_constructors(self) -> None:
        """Lazily initialize sub-constructors to avoid circular dependencies and registry issues."""
        if self._initialized:
            return

        from ...base.factory import RouteConstructorFactory

        # resolve names from params
        constructor_names = self.params.constructors

        self.constructors = [RouteConstructorFactory.get_adapter(name) for name in constructor_names]
        self._initialized = True

    def execute(
        self,
        **kwargs: Any,
    ) -> Tuple[Union[List[int], List[List[int]]], float, Optional[SearchContext]]:
        """
        Execute the sequence of route constructors.

        Args:
            **kwargs: Standard simulation context.

        Returns:
            Final (tour, cost, search_context) after all phases.
        """
        self._initialize_constructors()

        start_time = time.perf_counter()

        # Initialize with incoming state or valid default empty tour
        current_tour: Union[List[int], List[List[int]]] = kwargs.get("tour", [0, 0])
        current_cost: float = kwargs.get("cost", 0.0)
        current_context: Optional[SearchContext] = kwargs.get("search_context")

        for i, constructor in enumerate(self.constructors):
            # Check time limit
            elapsed = time.perf_counter() - start_time
            if elapsed > self.params.time_limit:
                logger.warning(
                    f"SRC time limit reached ({elapsed:.2f}s > {self.params.time_limit}s). "
                    f"Aborting sequence after {i} constructors."
                )
                break

            # Update kwargs to thread the state forward
            kwargs["tour"] = current_tour
            kwargs["cost"] = current_cost
            kwargs["search_context"] = current_context

            current_tour, current_cost, current_context = constructor.execute(**kwargs)

        return current_tour, current_cost, current_context

    def _run_solver(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Any,
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Not used as execute() is overridden directly."""
        return [], 0.0, 0.0
