r"""Stage 6 — Set-Partitioning merge (re-exported from pipeline_policy.stage_sp).

Thin re-export so the LBBD dispatcher imports a consistent package root.

Attributes:
    run_sp_stage: SP-merge entry point.
"""

from __future__ import annotations

from logic.src.policies.route_construction.pipelines.pipeline_policy.stage_sp import (
    run_sp_stage,
)

__all__ = ["run_sp_stage"]
