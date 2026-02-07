from .cvrpp import CVRPPState
from .env import EnvState
from .swcvrp import SWCVRPState
from .vrpp import VRPPState
from .wcvrp import WCVRPState

STATE_EMBEDDING_REGISTRY = {
    "vrpp": VRPPState,
    "cvrpp": CVRPPState,
    "wcvrp": WCVRPState,
    "cwcvrp": WCVRPState,
    "sdwcvrp": WCVRPState,
    "swcvrp": SWCVRPState,
    "scwcvrp": SWCVRPState,
}

__all__ = [
    "EnvState",
    "VRPPState",
    "WCVRPState",
    "CVRPPState",
    "SWCVRPState",
]
