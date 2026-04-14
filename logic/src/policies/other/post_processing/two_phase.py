"""
Two-Phase Post-Processor.
"""

from typing import Any, List

from logic.src.interfaces.post_processing import IPostProcessor

from .base import PostProcessorRegistry


@PostProcessorRegistry.register("two_phase")
class TwoPhasePostProcessor(IPostProcessor):
    """
    Composition post-processor that runs two separate strategies in sequence.
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
        # but factory.py uses Registry, so we can just use PostProcessorRegistry.

        phase_one_name = kwargs.get("phase_one", self.config.get("phase_one", "cheapest_insertion"))
        phase_two_name = kwargs.get("phase_two", self.config.get("phase_two", "lkh"))

        # 1. Phase One
        p1_cls = PostProcessorRegistry.get_post_processor_class(phase_one_name)
        if p1_cls is None:
            raise ValueError(f"Unknown phase_one strategy: {phase_one_name!r}")

        # Construct sub-processor with reference to full config dict
        p1 = p1_cls(config=self.config)
        tour = p1.process(tour, **kwargs)

        # 2. Phase Two
        p2_cls = PostProcessorRegistry.get_post_processor_class(phase_two_name)
        if p2_cls is None:
            raise ValueError(f"Unknown phase_two strategy: {phase_two_name!r}")

        p2 = p2_cls(config=self.config)
        tour = p2.process(tour, **kwargs)

        return tour
