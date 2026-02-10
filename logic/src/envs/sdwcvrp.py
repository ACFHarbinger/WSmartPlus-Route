"""
SDWCVRP Environment implementation.
"""

from __future__ import annotations

from tensordict import TensorDict

from logic.src.envs.wcvrp import WCVRPEnv


class SDWCVRPEnv(WCVRPEnv):
    """Stochastic Demand WCVRP with uncertain fill rates."""

    name: str = "sdwcvrp"

    def _step(self, td: TensorDict) -> TensorDict:
        """Step with stochastic demand simulation."""
        td = super()._step(td)

        # Simulate fill rate increase for unvisited bins
        if "fill_rates" in td:
            fill_increase = td["fill_rates"] * (~td["visited"]).float()
            td["waste"] = (td["waste"] + fill_increase).clamp(max=1.0)

        return td
