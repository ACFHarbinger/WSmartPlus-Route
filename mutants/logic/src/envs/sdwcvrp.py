"""
SDWCVRP Environment implementation.
"""

from __future__ import annotations

from logic.src.envs.wcvrp import WCVRPEnv
from tensordict import TensorDict


class SDWCVRPEnv(WCVRPEnv):
    """Stochastic Demand WCVRP with uncertain fill rates."""

    name: str = "sdwcvrp"

    def _step(self, td: TensorDict) -> TensorDict:
        """Step with stochastic demand simulation."""
        td = super()._step(td)

        # Simulate fill rate increase for unvisited bins
        if "fill_rates" in td.keys():
            fill_increase = td["fill_rates"] * (~td["visited"]).float()
            td["demand"] = (td["demand"] + fill_increase).clamp(max=1.0)

        return td
