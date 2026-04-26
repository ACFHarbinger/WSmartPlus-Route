r"""Stage 3 — ALNS (re-exported from pipeline_policy.stage_alns).

The LBBD pipeline reuses the identical ALNS stage from the TCF pipeline.
This thin module re-exports ``run_alns_stage`` so that the dispatcher can
import everything from a single package root.

Attributes:
    run_alns_stage: Pool-harvesting ALNS entry point.
"""

from __future__ import annotations

# Re-export without change — the implementation in pipeline_policy already
# has route-pool harvesting, warm-start injection, and ALNSParams patching.
from logic.src.policies.route_construction.pipelines.pipeline_policy.stage_alns import (
    run_alns_stage,
)

__all__ = ["run_alns_stage"]
