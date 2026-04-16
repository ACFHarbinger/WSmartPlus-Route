"""
IRouteConstructor — Route Construction Interface.

Defines the adapter contract for all routing policies (Neural, Classical,
Heuristic) that generate a tour from a mandatory bin list.  The third
element of the return tuple carries the ``SearchContext`` produced or
enriched during Phase 2 of the pipeline.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from logic.src.policies.context.search_context import SearchContext


class IRouteConstructor(ABC):
    """
    Interface for all routing policy adapters.

    Adapts various policies (Neural, Classical, Heuristic) to a common
    execution interface for the simulator.  The ``execute()`` method
    returns the ``SearchContext`` enriched during Phase 2 in the third
    slot of the tuple, allowing callers to thread it into Phase 3
    without invasive API changes.
    """

    @abstractmethod
    def execute(
        self,
        **kwargs: Any,
    ) -> Tuple[Union[List[int], List[List[int]]], float, Optional["SearchContext"]]:
        """
        Execute the policy to generate a route.

        Args:
            **kwargs: Context dictionary containing simulation state.
                May include ``search_context`` (a ``SearchContext`` from
                Phase 1) to carry forward the ledger.

        Returns:
            Tuple containing:
                - ``tour``: Flat or nested list of bin IDs.
                - ``cost``: Total routing cost.
                - ``search_context``: Updated ``SearchContext`` or ``None``
                  if context tracking is not used by this adapter.
        """
