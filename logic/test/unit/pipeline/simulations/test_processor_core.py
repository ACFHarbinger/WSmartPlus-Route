import pytest
import pandas as pd
import numpy as np
from logic.src.pipeline.simulations.processor import (
    setup_df,
    process_model_data,
)

class TestProcessor:
    """Class for data processing tests."""

    @pytest.mark.unit
    def test_setup_df(self):
        """Test DataFrame initialization for simulation."""
        depot = pd.DataFrame({"ID": [0], "Lat": [0.0], "Lng": [0.0], "Stock": [0], "Accum_Rate": [0]})
        df = pd.DataFrame({"ID": [1, 2], "Lat": [0.1, 0.2], "Lng": [0.1, 0.2]})
        col_names = ["ID", "Lat", "Lng"]
        process_df = setup_df(depot, df, col_names)
        assert isinstance(process_df, pd.DataFrame)
        assert "ID" in process_df.columns
        assert len(process_df) == 3

    @pytest.mark.unit
    def test_process_model_data(self, mock_sim_dependencies, mock_torch_device):
        """Test high-level data processing for model inputs."""
        # mock_sim_dependencies provides necessary patches
        coords = pd.DataFrame({
            "ID": [0, 1],
            "Lat": [40.0, 40.1],
            "Lng": [-8.0, -8.1]
        })
        dm = np.array([[0, 10], [10, 0]])
        configs = {"model": "am"}
        # Use a valid normalization method like "mmn"
        res = process_model_data(
            coords, dm, mock_torch_device, "mmn", configs, 50, "knn", "riomaior", "paper"
        )
        assert res is not None
