
import numpy as np
import pytest
from logic.src.policies.other.must_go.selection_last_minute import LastMinuteSelection
from logic.src.policies.other.must_go.selection_service_level import ServiceLevelSelection
from logic.src.policies.other.must_go.selection_revenue import RevenueThresholdSelection
from logic.src.policies.other.must_go.base.selection_context import SelectionContext

@pytest.fixture
def base_context():
    return SelectionContext(
        bin_ids=np.arange(1, 6),
        current_fill=np.array([10.0, 95.0, 30.0, 85.0, 50.0]),
        threshold=90.0,
        vehicle_capacity=100.0,
        bin_volume=1.0,
        bin_density=100.0,
        revenue_kg=2.0
    )

class TestLastMinuteSelection:
    def test_select_bins(self, base_context):
        strategy = LastMinuteSelection()
        must_go = strategy.select_bins(base_context)
        # Bins > 90 are Bin 2 (index 1) which is 95.
        assert must_go == [2]

    def test_select_bins_low_threshold(self, base_context):
        base_context.threshold = 40.0
        strategy = LastMinuteSelection()
        must_go = strategy.select_bins(base_context)
        # Bins > 40 are indices 1, 3, 4 -> IDs 2, 4, 5
        assert must_go == [2, 4, 5]

class TestServiceLevelSelection:
    def test_select_bins_no_data(self, base_context):
        strategy = ServiceLevelSelection()
        assert strategy.select_bins(base_context) == []

    def test_select_bins_with_data(self, base_context):
        base_context.accumulation_rates = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        base_context.std_deviations = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        base_context.threshold = 2.0 # Conf limit k=2
        # predicted = fill + means + k*std
        # Bin 4: 85 + 5 + 2*1 = 92 (No)
        # Bin 2: 95 + 5 + 2*1 = 102 (Yes)
        # Let's make Bin 4 also exceed 100
        base_context.current_fill[3] = 94
        # Bin 4: 94 + 5 + 2*1 = 101 (Yes)

        strategy = ServiceLevelSelection()
        must_go = strategy.select_bins(base_context)
        assert 2 in must_go
        assert 4 in must_go
        assert len(must_go) == 2

class TestRevenueThresholdSelection:
    def test_select_bins(self, base_context):
        # cap = vol * density = 1 * 100 = 100kg
        # rev = (fill/100) * cap * rate = fill * rate
        # Bins: [10*2, 95*2, 30*2, 85*2, 50*2] = [20, 190, 60, 170, 100]
        # Threshold 150
        base_context.threshold = 150.0
        strategy = RevenueThresholdSelection()
        must_go = strategy.select_bins(base_context)
        # Expected: Bin 2 (190) and Bin 4 (170)
        assert must_go == [2, 4]
