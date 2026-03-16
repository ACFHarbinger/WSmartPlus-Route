"""
Old Bachelor Acceptance (OBA) for VRPP.

OBA introduces a dynamically oscillating, non-monotonic acceptance threshold.
The threshold dilates after consecutive rejections (facilitating escape from
local optima) and contracts after consecutive acceptances (intensifying
exploitation of promising basins).

Reference:
    Hu, T. C., Kahng, A. B., & Tsao, C. A. "Old Bachelor Acceptance:
    A New Class of Non-Monotone Threshold Accepting Methods", 1995.
"""

from ..base.base_acceptance_criteria import BaseAcceptanceSolver


class OBASolver(BaseAcceptanceSolver):
    """
    Old Bachelor Acceptance solver for VRPP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OBA threshold starts at 0 (only strict improvements accepted initially)
        self.threshold = 0.0

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        """
        Accept if within threshold of current. Update threshold on acceptance/rejection.
        """
        if new_profit >= current_profit - self.threshold:
            # Contract threshold on acceptance
            self.threshold = max(0.0, self.threshold - self.params.contraction)
            return True
        else:
            # Dilate threshold on rejection
            self.threshold += self.params.dilation
            return False

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            threshold=self.threshold,
        )
