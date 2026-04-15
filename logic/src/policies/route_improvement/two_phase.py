"""
Two-Phase Route Improver.
"""

from typing import Any, List

from logic.src.interfaces.route_improvement import IRouteImprovement

from .base import RouteImproverRegistry


@RouteImproverRegistry.register("two_phase")
class TwoPhaseRouteImprover(IRouteImprovement):
    """
    Composition route improver that runs two separate strategies in sequence.
    Example: Augmentation phase followed by Improvement phase.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Run two-phase refinement on the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'phase_one', 'phase_two', and
                     parameters for both phases.

        Returns:
            List[int]: Refined tour after both phases.

        Note:
            Both phases share the same `self.config` dictionary. Phase-specific
            overrides (e.g., using different `cost_per_km` for each phase)
            are not supported in this compositional strategy.
        """
        # Lazy import of factory to avoid circular dependencies if needed,
        # but factory.py uses Registry, so we can just use RouteImproverRegistry.

        phase_one_name = kwargs.get("phase_one", self.config.get("phase_one", "cheapest_insertion"))
        phase_two_name = kwargs.get("phase_two", self.config.get("phase_two", "lkh"))

        # 1. Phase One
        p1_cls = RouteImproverRegistry.get_route_improver_class(phase_one_name)
        if p1_cls is None:
            raise ValueError(f"Unknown phase_one strategy: {phase_one_name!r}")

        # Construct sub-processor with reference to full config dict
        p1 = p1_cls(config=self.config)
        tour = p1.process(tour, **kwargs)

        # 2. Phase Two
        p2_cls = RouteImproverRegistry.get_route_improver_class(phase_two_name)
        if p2_cls is None:
            raise ValueError(f"Unknown phase_two strategy: {phase_two_name!r}")

        p2 = p2_cls(config=self.config)
        tour = p2.process(tour, **kwargs)

        return tour
