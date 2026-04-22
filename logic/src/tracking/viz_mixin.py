"""
Policy Visualization Mixin.

Provides lightweight, zero-overhead state recording for classical and
heuristic policies.  Import :class:`PolicyVizMixin` and add it as a
mixin base-class to any policy that should expose iteration data to the
Streamlit dashboard.


Attributes:
    PolicyStateRecorder: Fixed-capacity ring-buffer for per-iteration policy telemetry.
    PolicyVizMixin: Mixin that adds zero-overhead telemetry to policy classes.

Example:

    class VectorizedALNS(PolicyVizMixin):
        def solve(self, ...):
            for i in range(n_iterations):
                ...
                self._viz_record(iteration=i, best_cost=best_costs.min().item())

    # After running:
    data = solver.get_viz_data()
    # data["iteration"], data["best_cost"]  → lists of recorded values
"""

import contextlib
from typing import Any, Dict, List


class PolicyStateRecorder:
    """
    Fixed-capacity ring-buffer for per-iteration policy telemetry.

    Attributes:
        _buf: Dictionary for storing telemetry data.
        _max: Maximum number of records to keep.
    """

    def __init__(self, max_history: int = 5000) -> None:
        """
        Initialize the PolicyStateRecorder.

        Args:
            max_history: Maximum number of records to keep.  Oldest entries
                         are silently dropped once the buffer is full.

        Returns:
            None
        """
        self._buf: Dict[str, List[Any]] = {}
        self._max = max_history

    def record(self, **kwargs: Any) -> None:
        """
        Append one snapshot of keyword-named scalar metrics.

        Args:
            kwargs: Keyword arguments representing the metrics to record.

        Returns:
            None
        """
        for key, val in kwargs.items():
            lst = self._buf.setdefault(key, [])
            lst.append(val)
            if len(lst) > self._max:
                lst.pop(0)

    def get(self) -> Dict[str, List[Any]]:
        """
        Return a shallow copy of the accumulated telemetry.

        Returns:
            Dict[str, List[Any]]: A dictionary containing the accumulated telemetry.
        """
        return {k: list(v) for k, v in self._buf.items()}

    def reset(self) -> None:
        """
        Discard all recorded data.

        Returns:
            None
        """
        self._buf.clear()

    def __len__(self) -> int:
        """
        Return the number of records.

        Returns:
            int: The number of records.
        """
        if not self._buf:
            return 0
        return max(len(v) for v in self._buf.values())


class PolicyVizMixin:
    """
    Mixin that adds zero-overhead telemetry to policy classes.

    The internal :class:`PolicyStateRecorder` is only created on the
    first call to :meth:`_viz_record`, so policies that never record
    data pay no allocation cost at all.

    Concrete policies should call ``self._viz_record(**metrics)`` inside
    their main iteration loop, then consumers call
    ``policy.get_viz_data()`` to retrieve the captured time-series.

    Attributes:
        _viz_recorder: PolicyStateRecorder for storing telemetry data.
    """

    # ------------------------------------------------------------------
    # Lazy recorder property
    # ------------------------------------------------------------------

    @property
    def _viz(self) -> PolicyStateRecorder:
        """
        Lazy property for the recorder.

        Returns:
            PolicyStateRecorder: The recorder.
        """
        try:
            return self.__viz_recorder  # type: ignore[attr-defined]
        except AttributeError:
            self.__viz_recorder: PolicyStateRecorder = PolicyStateRecorder()
            return self.__viz_recorder

    # ------------------------------------------------------------------
    # Public API (forwarded to recorder)
    # ------------------------------------------------------------------

    def _viz_record(self, **kwargs: Any) -> None:
        """
        Record one snapshot of metrics. Call once per iteration.

        Args:
            kwargs: Keyword arguments representing the metrics to record.

        Returns:
            None
        """
        self._viz.record(**kwargs)

    def get_viz_data(self) -> Dict[str, List[Any]]:
        """
        Return accumulated telemetry as a plain dict of lists,
        or an empty dict if ``_viz_record`` was never called.

        Returns:
            Dict[str, List[Any]]: A dictionary containing the accumulated telemetry.
        """
        try:
            return self.__viz_recorder.get()  # type: ignore[attr-defined]
        except AttributeError:
            return {}

    def reset_viz(self) -> None:
        """
        Clear all recorded telemetry (safe to call before re-use).

        Returns:
            None
        """
        with contextlib.suppress(AttributeError):
            self.__viz_recorder.reset()  # type: ignore[attr-defined]
