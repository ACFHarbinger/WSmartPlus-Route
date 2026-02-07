"""Tests for the Container class in wsmart_bin_analysis."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from logic.src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container import TAG, Container


class TestContainer:
    """Test suite for the Container class."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for container tests.

        Returns:
            tuple: (df, recs, info) for initialized Container.
        """
        # Create sample DataFrames
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        fill_data = {"Date": dates, "Fill": np.linspace(0, 100, 10), "ID": [1] * 10}
        df = pd.DataFrame(fill_data)

        # Collections on day 0, 5, 9
        rec_dates = [dates[0], dates[5], dates[9]]
        rec_data = {
            "Date": rec_dates,
            "ID": [1] * 3,
            "Avg_Dist": [90, 80, 95],  # Dummy values
            "Spearman": [0.9, 0.8, 0.95],
        }
        recs = pd.DataFrame(rec_data)

        info = pd.DataFrame({"ID": [1], "Freguesia": ["Test"]})

        return df, recs, info

    def test_init(self, sample_data):
        """Test initialization of Container."""
        df, recs, info = sample_data
        container = Container(df.copy(), recs.copy(), info.copy())

        assert container.id.item() == 1
        assert "Fill" in container.df.columns
        assert "Avg_Dist" in container.recs.columns
        assert container.tag is None

    def test_get_keys_and_vars(self, sample_data):
        """Test retrieval of keys and variables."""
        df, recs, info = sample_data
        container = Container(df, recs, info)

        keys = container.get_keys()
        assert "FILL" in keys
        assert "RECS" in keys

        d, r, i = container.get_vars()
        assert d.equals(container.df)
        assert r.equals(container.recs)

    def test_get_collection_quantities(self, sample_data):
        """Test calculation of avg distance and spearman."""
        df, recs, info = sample_data
        container = Container(df, recs, info)

        avg_dist, spear = container.get_collection_quantities()
        assert len(avg_dist) == 3
        assert len(spear) == 3
        assert avg_dist[0] == 90

    def test_mark_collections(self, sample_data):
        """Test marking of collections in dataframe."""
        df, recs, info = sample_data
        container = Container(df, recs, info)

        container.mark_collections()
        assert "Rec" in container.df.columns
        assert "Cidx" in container.df.columns
        # Check if collection marks are set (Rec=1 on collection days usually, or logic inside)
        # Logic: df["Rec"] = 1 where pos matches
        assert container.df["Rec"].sum() >= 3

    def test_calc_max_min_mean(self, sample_data):
        """Test calculation of max, min, and mean fill levels."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()

        container.calc_max_min_mean()
        assert "Max" in container.df.columns
        assert "Min" in container.df.columns
        assert "Mean" in container.df.columns

        # Basic check: Min <= Max
        assert (container.df["Min"] <= container.df["Max"]).all()

    def test_calc_avg_dist_metric(self, sample_data):
        """Test calculation of average distance metrics."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()
        container.calc_max_min_mean()

        container.calc_avg_dist_metric()
        assert "Avg_Dist" in container.recs.columns
        # Since we overwrote it, check it exists and has values
        assert not container.recs["Avg_Dist"].isna().all()

    def test_calc_spearman(self, sample_data):
        """Test calculation of Spearman correlation for fill rates."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()

        container.calc_spearman()
        assert "Spearman" in container.recs.columns
        # Check values
        assert not container.recs["Spearman"].isna().all()

    def test_get_tag_ok(self, sample_data):
        """Test successful tagging of container quality."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        # Ensure enough data
        # min_days
        tag = container.get_tag(window=2, mv_thresh=0.5, min_days=2, use="spear")
        # With high spearman values (0.9), it should be OK or INSIDE_BOX depending on transitions
        assert isinstance(tag, TAG)

    def test_get_tag_low_measures(self):
        """Test tagging when there are too few measures."""
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        df = pd.DataFrame({"Date": dates, "Fill": [10], "ID": [1]})
        recs = pd.DataFrame({"Date": dates, "ID": [1], "Spearman": [0]})
        info = pd.DataFrame({"ID": [1], "Freguesia": ["Test"]})

        container = Container(df, recs, info)
        tag = container.get_tag(window=1, mv_thresh=0.5, min_days=5, use="spear")
        assert tag == TAG.LOW_MEASURES

    def test_get_scan_linear_spline(self, sample_data):
        """Test linear spline interpolation for scans."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()
        container.calc_max_min_mean()

        dates, spline = container.get_scan_linear_spline("Mean", "1D")
        assert len(dates) == len(spline)

    def test_get_monotonic_mean_rate(self, sample_data):
        """Test calculation of monotonic mean fill rate."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()
        container.calc_max_min_mean()

        rate_df = container.get_monotonic_mean_rate("1D")
        assert "Rate" in rate_df.columns
        assert len(rate_df) > 0

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_fill(self, mock_fig, mock_show, sample_data):
        """Test plotting of container fill levels."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()

        start = "01-01-2023"
        end = "05-01-2023"
        container.plot_fill(start, end)
        assert mock_fig.called
        assert mock_show.called

    def test_adjust_collections(self, sample_data):
        """Test adjusting collections based on distance thresholds."""
        df, recs, info = sample_data
        # Make a case where Avg_Dist is low to trigger adjustment
        recs["Avg_Dist"] = 10.0  # Low threshold

        container = Container(df, recs, info)
        container.mark_collections()
        container.calc_max_min_mean()

        # Should try to adjust/delete collections
        try:
            container.adjust_collections(dist_thresh=20, c_trash=0, max_fill=200)
        except Exception as e:
            pytest.fail(f"adjust_collections failed: {e}")

    def test_place_collections(self, sample_data):
        """Test placing collections based on fill levels and thresholds."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()
        # Mock low dist to trigger placement check
        container.recs["Avg_Dist"] = 10.0

        try:
            container.place_collections(dist_thresh=20, c_trash=0, max_fill=200)
        except Exception as e:
            pytest.fail(f"place_collections failed: {e}")

    def test_clean_box(self, sample_data):
        """Test cleaning/filtering of record quality."""
        df, recs, info = sample_data
        container = Container(df, recs, info)
        container.mark_collections()  # Needed to set End_Pointer

        # Mixed quality: One good (0.9), others bad (0.1)
        # recs len is 3.
        # Set explicitly.
        container.recs["Spearman"] = [0.1, 0.9, 0.1]

        # Should filter out the 0.1 ones.
        # Threshold 0.5.
        # At least one should survive.
        container.clean_box(window=1, mv_thresh=0.5, use="spear")

        # Expecting at least 1 record remaining
        assert len(container.recs) > 0
        assert "End_Pointer" in container.recs.columns
