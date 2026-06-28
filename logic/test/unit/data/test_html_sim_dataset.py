"""Tests for HTML simulation dataset and crawler."""

import os
import tempfile

import numpy as np
import pytest
from logic.src.data.datasets.web.dashboard_crawler import extract_dataframe
from logic.src.data.datasets.web.html_sim_dataset import HtmlSimulationDataset

MOCK_HTML = """
<div class="flow-section">
    <div class="tab-bar">
        <button>Todos os Locais</button>
    </div>
    <div class="tab-content">
        <table>
            <tr>
                <th>Local</th>
                <th>% vol. atual</th>
                <th>% vol. média</th>
                <th>Acum. (%/dia)</th>
                <th>Volume (kg)</th>
                <th>Nº Cont.</th>
                <th>Latitude</th>
                <th>Longitude</th>
                <th>Viagem / Origem</th>
            </tr>
            <tr>
                <td>101</td>
                <td>35.5%</td>
                <td>40.0%</td>
                <td>12.5</td>
                <td>150.0</td>
                <td>2</td>
                <td>39.1887</td>
                <td>-9.1484</td>
                <td>Route A</td>
            </tr>
            <tr>
                <td>102</td>
                <td>80.0%</td>
                <td>75.0%</td>
                <td>8.0</td>
                <td>100.0</td>
                <td>1</td>
                <td>39.1921</td>
                <td>-9.1523</td>
                <td>Route B</td>
            </tr>
        </table>
    </div>
</div>
"""


class TestHtmlSimulationDataset:
    """Tests for HtmlSimulationDataset and dashboard crawler."""

    @pytest.fixture
    def mock_html_file(self):
        """Create a temporary HTML file containing the mock dashboard."""
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
            f.write(MOCK_HTML)
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_extract_dataframe(self, mock_html_file):
        """Test crawler table extraction."""
        df = extract_dataframe(mock_html_file)
        assert len(df) == 2
        assert list(df["ID"]) == [101, 102]
        assert list(df["Fill_Pct"]) == [35.5, 80.0]
        assert list(df["Acum_Rate_Pct"]) == [12.5, 8.0]
        assert list(df["Volume_kg"]) == [150.0, 100.0]
        assert list(df["N_Containers"]) == [2, 1]
        assert list(df["Lat"]) == [39.1887, 39.1921]
        assert list(df["Lng"]) == [-9.1484, -9.1523]

    def test_html_sim_dataset_load(self, mock_html_file):
        """Test loading HTML simulation dataset and checking fields."""
        dataset = HtmlSimulationDataset.load(mock_html_file, n_days=5)
        assert len(dataset) == 1

        sample = dataset[0]
        assert "depot" in sample
        assert "locs" in sample
        assert "waste" in sample
        assert "noisy_waste" in sample
        assert "max_waste" in sample
        assert "node_ids" in sample

        # Check shapes
        assert sample["locs"].shape == (2, 2)
        assert sample["waste"].shape == (5, 2)  # n_days=5, n_bins=2
        assert sample["noisy_waste"].shape == (5, 2)
        assert sample["max_waste"].shape == (2,)

        # Check values
        # Day 0: [35.5, 80.0]
        # Day 1: [35.5 + 12.5, 80.0 + 8.0] = [48.0, 88.0]
        # Day 2: [60.5, 96.0]
        # Day 3: [73.0, 100.0] (clipped)
        np.testing.assert_allclose(sample["waste"][0], [35.5, 80.0])
        np.testing.assert_allclose(sample["waste"][1], [48.0, 88.0])
        np.testing.assert_allclose(sample["waste"][2], [60.5, 96.0])
        np.testing.assert_allclose(sample["waste"][3], [73.0, 100.0])
        np.testing.assert_allclose(sample["waste"][4], [85.5, 100.0])

    def test_html_sim_dataset_indexing(self, mock_html_file):
        """Verify dataset indexing rules."""
        dataset = HtmlSimulationDataset.load(mock_html_file, n_days=3)
        with pytest.raises(IndexError):
            _ = dataset[1]
