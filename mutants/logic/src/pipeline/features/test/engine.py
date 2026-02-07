"""
Simulation testing engine for WSmart-Route.

This module is now a facade for `logic.src.pipeline.features.test.engine`.
"""

from logic.src.pipeline.features.test.engine import (
    run_wsr_simulator_test,
    simulator_testing,
)

__all__ = ["simulator_testing", "run_wsr_simulator_test"]
