"""
IRouteConstructor — Route Construction Interface.

Defines the adapter contract for all routing policies (Neural, Classical,
Heuristic) that generate a tour from a mandatory bin list.  The third
element of the return tuple carries the ``SearchContext`` produced or
enriched during Phase 2 of the pipeline.

Attributes:
    IRouteConstructor: Interface for all routing route constructors

Example:
    >>> from logic.src.interfaces.route_constructor import IRouteConstructor
    >>> from logic.src.interfaces.context.search_context import SearchContext
    >>> from logic.src.interfaces.context.multi_day_context import MultiDayContext
    >>> class MyRouteConstructor(IRouteConstructor):
    ...     def execute(
    ...         self,
    ...         **kwargs: Any,
    ...     ) -> Tuple[
    ...         Union[List[int], List[List[int]]],
    ...         float,
    ...         float,
    ...         Optional["SearchContext"],
    ...         Optional["MultiDayContext"],
    ...     ]:
    ...         return [], 0.0, 0.0, None, None
    ...
    >>> route_constructor = MyRouteConstructor()
    >>> route_constructor.execute()
    ([], 0.0, 0.0, None, None)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from logic.src.interfaces.context.multi_day_context import MultiDayContext
    from logic.src.interfaces.context.search_context import SearchContext


class IRouteConstructor(ABC):
    """
    Interface for all routing policy adapters.

    Adapts various policies (Neural, Classical, Heuristic) to a common
    execution interface for the simulator.  The ``execute()`` method
    returns the ``SearchContext`` enriched during Phase 2 in the fourth
    slot, profit in the third slot, and optionally a ``MultiDayContext``
    in the fifth slot for rolling horizon state tracking.

    Attributes:
        None: No attributes
    """

    @abstractmethod
    def execute(
        self,
        **kwargs: Any,
    ) -> Tuple[
        Union[List[int], List[List[int]]],
        float,
        float,
        Optional["SearchContext"],
        Optional["MultiDayContext"],
    ]:
        """
        Execute the policy to generate a route.

        Args:
            kwargs: Context dictionary containing simulation state.
                May include ``search_context`` (a ``SearchContext`` from
                Phase 1) and ``multi_day_context`` (historical data).

        Returns:
            Tuple containing:
                - ``tour``: Flat or nested list of bin IDs.
                - ``cost``: Total routing cost.
                - ``profit``: Net profit (revenue - cost).
                - ``search_context``: Updated ``SearchContext``.
                - ``multi_day_context``: Rolling horizon state metadata.
        """
