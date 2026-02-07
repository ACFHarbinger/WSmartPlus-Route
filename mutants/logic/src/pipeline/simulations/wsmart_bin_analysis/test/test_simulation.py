"""
Unit tests for the GridBase and Simulation classes in the wsmart_bin_analysis module.
"""

import numpy as np
import pandas as pd

from ..Deliverables.simulation import GridBase, Simulation


class TestGridBase:
    """
    Test suite for the GridBase class, which manages a collection of container data.
    """

    def test_gridbase_init(self, mock_load_data):
        """Test that GridBase initializes correctly and reports the correct number of bins."""
        mock_rate, mock_info, mock_wrapper = mock_load_data

        # Mock return values
        dates = pd.date_range("2020-01-01", periods=10)
        data_df = pd.DataFrame(np.random.rand(10, 2), index=dates, columns=[1, 2])
        mock_wrapper.return_value = data_df

        mock_info.return_value = {"ID": 1, "Lat": 0, "Lon": 0}

        gb = GridBase(ids=[1, 2], data_dir="/tmp", rate_type="mean")

        assert gb.data is not None
        assert gb.get_num_bins() == 2
        assert gb.get_datarange() == (dates[0], dates[-1])


class TestSimulation:
    """
    Test suite for the Simulation class, covering timestep progression and collection logic.
    """

    def test_simulation_step(self, mock_load_data):
        """Test advancing the simulation by one timestep and handling overflows."""
        mock_rate, mock_info, mock_wrapper = mock_load_data

        dates = pd.date_range("2020-01-01", periods=10)
        data_df = pd.DataFrame(10 * np.ones((10, 1)), index=dates, columns=[1])
        mock_wrapper.return_value = data_df
        mock_info.return_value = {"ID": 1}

        sim = Simulation(
            sim_type="real",
            ids=[1],
            data_dir="/tmp",
            start_date="01-01-2020",
            end_date="05-01-2020",
            rate_type="mean",
            predictQ=False,
        )

        # Initial state
        assert sim.fill[0] == 0

        # Step 1
        n_over, pred, err = sim.advance_timestep()
        assert sim.fill[0] == 10

        # Check overflow
        sim.fill[0] = 95
        sim.advance_timestep()
        assert sim.fill[0] == 100
        n_over, _, _ = sim.advance_timestep()
        assert n_over > 0

    def test_make_collections(self, mock_load_data):
        """Test the logic for recording a collection event and resetting bin fill level."""
        mock_rate, mock_info, mock_wrapper = mock_load_data
        dates = pd.date_range("2020-01-01", periods=5)
        data_df = pd.DataFrame(10 * np.ones((5, 1)), index=dates, columns=[1])
        mock_wrapper.return_value = data_df

        sim = Simulation(
            sim_type="real",
            ids=[1],
            data_dir="/tmp",
            start_date="01-01-2020",
            end_date="03-01-2020",
            rate_type="mean",
            predictQ=False,
        )

        sim.fill[0] = 50
        collected = sim.make_collections(bins_index_list=[0])

        assert collected[0] == 50
        assert sim.fill[0] == 0
