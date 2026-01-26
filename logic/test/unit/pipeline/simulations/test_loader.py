from unittest.mock import mock_open, patch

import logic.src.constants as udef
import pandas as pd
import pytest
from logic.src.pipeline.simulations.loader import (
    FileSystemRepository,
    _repository,
    load_area_and_waste_type_params,
)


class TestFileSystemRepository:
    @pytest.fixture
    def repo(self, tmp_path):
        return FileSystemRepository(str(tmp_path))

    def test_get_area_params(self, repo):
        # Test known params
        # Rio Maior, Paper
        # vehicle_capacity = 4000L. bin_vol=2.5. density=21.
        # capacity % = (4000 / (2.5 * 21)) * 100 = 7619.04...
        cap, rev, dens, exp, vol = repo.get_area_params("Rio Maior", "paper")

        assert dens == 21.0
        assert vol == 2.5
        assert exp == 1
        assert cap > 0

        # Test error
        with pytest.raises(AssertionError):
            repo.get_area_params("Rio Maior", "uranium")

    def test_get_depot_mocked(self, repo):
        # Mock pd.read_csv to avoid file IO
        mock_facilities = pd.DataFrame({"Sigla": ["RM"], "Lat": [10.0], "Lng": [20.0]})

        with patch("pandas.read_csv", return_value=mock_facilities):
            # Udef mock needed for MAP_DEPOTS?
            # src_area "riomaior" -> MAP_DEPOTS["riomaior"] = "RM" usually
            with patch.dict(udef.MAP_DEPOTS, {"riomaior": "RM"}):
                depot = repo.get_depot("Rio Maior")

                assert len(depot) == 1
                assert depot.iloc[0]["Lat"] == 10.0
                assert "Stock" in depot.columns

    def test_get_indices_create_new(self, repo):
        # Mock file not exists, should create new
        # n_samples=2, n_nodes=3, data_size=10
        filename = "test_indices.json"

        # We need to mock open to verify write
        m_open = mock_open()

        with patch("os.path.isfile", return_value=False), patch("builtins.open", m_open), patch(
            "json.dumps", return_value="[1,2,3]"
        ):
            indices = repo.get_indices(filename, n_samples=2, n_nodes=3, data_size=10)

            assert len(indices) == 2
            assert len(indices[0]) == 3

            # Verify file write
            m_open.assert_called()
            handle = m_open()
            handle.write.assert_called()

    def test_get_indices_load_existing(self, repo):
        filename = "test_indices.json"
        mock_data = [[1, 2, 3], [4, 5, 6]]

        with patch("os.path.isfile", return_value=True), patch(
            "builtins.open", mock_open(read_data="[[1,2,3],[4,5,6]]")
        ), patch("json.load", return_value=mock_data):
            indices = repo.get_indices(filename, 2, 3, 10)
            assert indices == mock_data

    def test_wrappers(self):
        # Test that wrappers call the singleton _repository
        with patch.object(_repository, "get_area_params") as mock_method:
            load_area_and_waste_type_params("A", "B")
            mock_method.assert_called_with("A", "B")
