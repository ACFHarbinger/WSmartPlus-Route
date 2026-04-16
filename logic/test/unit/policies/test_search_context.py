"""
Unit tests for functional SearchContext tracking.
"""

import pytest
from logic.src.policies.context.search_context import (
    SearchContext,
    SearchPhase,
    merge_context,
    SelectionMetrics,
    ConstructionMetrics,
    ImprovementMetrics,
)


def test_search_context_initialization():
    """Verify context initialization."""
    sel_metrics: SelectionMetrics = {"strategy": "TestSelection"}
    ctx = SearchContext.initialize(selection_metrics=sel_metrics)

    assert ctx.selection["strategy"] == "TestSelection"
    assert ctx.construction is None
    assert ctx.improvement == []


def test_merge_construction():
    """Verify merging construction metrics (Phase 2)."""
    ctx = SearchContext.initialize(selection_metrics={"strategy": "Sel"})
    cons_metrics: ConstructionMetrics = {"algorithm": "Exact", "n_mandatory": 5, "profit": 100.0}

    new_ctx = merge_context(ctx, phase=SearchPhase.CONSTRUCTION, construction_metrics=cons_metrics)

    # Original should be untouched (immutable)
    assert ctx.construction is None
    # New should have metrics
    assert new_ctx.construction["algorithm"] == "Exact"
    assert new_ctx.construction["profit"] == 100.0


def test_merge_improvement():
    """Verify appending improvement metrics (Phase 3)."""
    ctx = SearchContext.initialize(selection_metrics={"strategy": "Sel"})
    imp_metrics: ImprovementMetrics = {"algorithm": "SA", "n_iterations": 10}

    new_ctx = merge_context(ctx, phase=SearchPhase.IMPROVEMENT, improvement_metrics=imp_metrics)

    assert len(ctx.improvement) == 0
    assert len(new_ctx.improvement) == 1
    assert new_ctx.improvement[0]["algorithm"] == "SA"

    # Append second improver
    imp_metrics2: ImprovementMetrics = {"algorithm": "2Opt"}
    final_ctx = merge_context(new_ctx, phase=SearchPhase.IMPROVEMENT, improvement_metrics=imp_metrics2)

    assert len(final_ctx.improvement) == 2
    assert final_ctx.improvement[1]["algorithm"] == "2Opt"


def test_full_flow():
    """Verify end-to-end telemetry flow."""
    # 1. Selection
    ctx = SearchContext.initialize({"strategy": "Learned"})

    # 2. Construction
    ctx = merge_context(
        ctx, SearchPhase.CONSTRUCTION, construction_metrics={"algorithm": "AM", "profit": 50.0}
    )

    # 3. Improvement
    ctx = merge_context(
        ctx, SearchPhase.IMPROVEMENT, improvement_metrics={"algorithm": "LS", "best_delta": -5.0}
    )

    assert ctx.selection["strategy"] == "Learned"
    assert ctx.construction["algorithm"] == "AM"
    assert len(ctx.improvement) == 1
    assert ctx.improvement[0]["algorithm"] == "LS"


def test_mismatched_metrics_raises():
    """Verify that passing wrong metrics for a phase raises ValueError."""
    ctx = SearchContext.initialize({"strategy": "Sel"})

    with pytest.raises(ValueError, match="construction_metrics must be provided"):
        merge_context(ctx, phase=SearchPhase.CONSTRUCTION)

    with pytest.raises(ValueError, match="improvement_metrics must be provided"):
        merge_context(ctx, phase=SearchPhase.IMPROVEMENT)
