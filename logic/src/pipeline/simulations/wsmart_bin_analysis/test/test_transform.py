"""
Unit tests for data transformation and fixing utilities in the wsmart_bin_analysis module.
"""

from ..Deliverables.container import TAG, Container
from ..Deliverables.transform import (
    filter_containers,
    fix_collections_sensor,
    get_overall_sensors_statistics,
)


class TestTransform:
    """
    Test suite for transformation functions that clean or adjust container data.
    """

    def test_fix_collections_sensor(self, sample_container):
        """Test the heuristic-based correction of sensor measurements and collection events."""
        c = sample_container
        c.mark_collections()
        c.calc_max_min_mean()
        c.calc_avg_dist_metric()
        c.calc_spearman()

        dvar, tags, new_c = fix_collections_sensor(
            container=c,
            box_window=3,
            mv_thresh=50,
            min_days=0,
            dist_thresh=20,
            c_trash=0,
            max_fill=100,
            var_thresh=10,
            use="spear",
        )

        assert isinstance(new_c, Container)
        assert len(tags) > 0
        assert isinstance(dvar, (int, float))

    def test_filter_containers(self, sample_container):
        """Test filtering a dictionary of containers based on their quality tags."""
        c = sample_container
        c.tag = TAG.OK
        d = {1: c}
        res = filter_containers(d)
        assert 1 in res

        c.tag = TAG.WARN
        res = filter_containers(d)
        assert 1 not in res

    def test_get_overall_sensors_statistics(self, sample_container):
        """Test aggregation of statistics (distance, spearman) across multiple containers."""
        c = sample_container
        c.mark_collections()
        c.calc_max_min_mean()
        c.calc_avg_dist_metric()
        c.calc_spearman()

        d = {1: c}
        dist, spear = get_overall_sensors_statistics(d)
        assert 1 in dist
        assert 1 in spear
