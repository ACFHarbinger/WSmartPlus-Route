r"""Stage 4 — BPC (re-exported from pipeline_policy.stage_bpc).

Thin re-export so the LBBD dispatcher imports a consistent package root.
The BPC stage implementation in pipeline_policy already handles incumbent
seeding, pool injection, and alpha-scaled ng-size / bb-node limits.

Attributes:
    run_bpc_stage: BPC entry point.
"""

from __future__ import annotations

from logic.src.policies.route_construction.pipelines.pipeline_policy.stage_bpc import (
    run_bpc_stage,
)

__all__ = ["run_bpc_stage"]
