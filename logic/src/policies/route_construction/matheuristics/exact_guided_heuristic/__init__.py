r"""TCF → ALNS → BPC → SP-merge pipeline package.

Public API
----------
PipelinePolicy   Policy adapter (registers as "pipeline" in the solver registry).
PipelineParams   Configuration dataclass with ``alpha`` quality/speed dial.
run_pipeline     Low-level orchestrator — can be used as a drop-in replacement
                 for ``_run_gurobi_optimizer``.

Quick start
-----------
Drop-in replacement for the bare SWC-TCF solver::

    from <package>.pipeline import run_pipeline, PipelineParams

    params = PipelineParams(alpha=0.5, time_limit=120)
    flat_route, profit, cost = run_pipeline(
        bins, dist_matrix, env, values, binsids, mandatory,
        n_vehicles=2, params=params,
    )

Policy registration (automatic on import)::

    from <package>.pipeline import PipelinePolicy   # registers "pipeline"
    policy = PipelinePolicy()
"""

from .dispatcher import run_pipeline
from .params import ExactGuidedHeuristicParams
from .policy_egh import ExactGuidedHeuristicPolicy
from .route_pool import RoutePool, VRPPRoute

__all__ = [
    "ExactGuidedHeuristicPolicy",
    "ExactGuidedHeuristicParams",
    "RoutePool",
    "VRPPRoute",
    "run_pipeline",
]
