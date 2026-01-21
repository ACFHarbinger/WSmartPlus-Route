"""
Unit tests for the Container class in the wsmart_bin_analysis module.
"""

import numpy as np
import pytest

from ..Deliverables.container import TAG


class TestContainer:
    """
    Test suite for the Container class, covering initialization, metric calculation, and collection marking.
    """

    def test_container_init(self, sample_container, sample_df_fill, sample_df_collection, sample_info):
        """Test that the Container initializes correctly and drops the ID column."""
        c = sample_container
        # Check if df and recs are set and ID dropped
        assert "ID" not in c.df.columns
        assert "ID" not in c.recs.columns
        assert c.id.iloc[0] == sample_info["ID"].iloc[0]

    def test_get_keys(self, sample_container):
        """Test the get_keys method returns the expected keys."""
        keys = sample_container.get_keys()
        assert "FILL" in keys
        assert "RECS" in keys
        assert "INFO" in keys

    def test_mark_collections(self, sample_container):
        """Test that mark_collections correctly identifies and tags collection events."""
        c = sample_container
        c.mark_collections()
        assert "Rec" in c.df.columns
        assert "Cidx" in c.df.columns
        assert "End_Pointer" in c.recs.columns

        # Check if collections are marked
        assert c.df["Rec"].sum() > 0

    def test_calc_metrics(self, sample_container):
        """Test calculation of various container metrics (max/min/mean, avg dist, spearman)."""
        c = sample_container
        c.mark_collections()

        # Run calcs
        c.calc_max_min_mean()
        assert "Max" in c.df.columns
        assert "Min" in c.df.columns
        assert "Mean" in c.df.columns

        c.calc_avg_dist_metric()
        assert "Avg_Dist" in c.recs.columns

        c.calc_spearman()
        assert "Spearman" in c.recs.columns

    def test_get_collection_quantities(self, sample_container):
        """Test retrieving collection-related statistics."""
        c = sample_container
        c.mark_collections()
        c.calc_max_min_mean()
        c.calc_avg_dist_metric()
        c.calc_spearman()

        avg_dist, spear = c.get_collection_quantities()
        assert isinstance(avg_dist, (np.ndarray, type(None)))
        assert isinstance(spear, (np.ndarray, type(None)))
        # With the fixture data, we should have some results if overlap existed
        if avg_dist is not None:
            assert len(avg_dist) <= len(c.recs)
            assert len(avg_dist) > 0

    def test_get_tag(self, sample_container):
        """Test assigning a quality tag to a container based on its data consistency."""
        c = sample_container
        c.mark_collections()
        c.calc_max_min_mean()
        c.calc_avg_dist_metric()
        c.calc_spearman()

        # Test valid use
        tag = c.get_tag(window=3, mv_thresh=50, min_days=1, use="spear")
        assert isinstance(tag, TAG)

        # Test invalid use
        with pytest.raises(AssertionError):
            c.get_tag(window=3, mv_thresh=50, min_days=1, use="invalid")
