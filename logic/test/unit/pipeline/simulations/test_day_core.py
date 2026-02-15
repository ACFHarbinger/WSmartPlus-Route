import pytest
import pandas as pd
from logic.src.pipeline.simulations.day_context import run_day

class TestDay:
    """Class for daily execution tests."""

    @pytest.mark.unit
    def test_run_day(self, make_day_context, mock_run_day_deps):
        """Test the run_day orchestrator."""
        context = make_day_context()
        # Ensure we have some actions to run
        context.new_data = mock_run_day_deps["new_data"]
        context.coords = mock_run_day_deps["coords"]
        context.bins = mock_run_day_deps["bins"]

        # Mocking actions to avoid complex logic checks here
        updated_context = run_day(context)
        assert updated_context.day == context.day
