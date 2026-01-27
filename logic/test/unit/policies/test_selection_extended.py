
import numpy as np
import pytest
from logic.src.policies.selection.selection_last_minute import LastMinuteSelection, LastMinuteAndPathSelection
from logic.src.policies.selection.selection_means_std import MeansAndStdDevSelection
from logic.src.policies.selection.selection_revenue import RevenueThresholdSelection
from logic.src.policies.must_go_selection import SelectionContext

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

class TestLastMinuteAndPathSelection:
    def test_select_bins_no_critical(self, base_context):
        base_context.current_fill = np.zeros(5)
        strategy = LastMinuteAndPathSelection()
        assert strategy.select_bins(base_context) == []

    def test_select_bins_no_path_info(self, base_context):
        strategy = LastMinuteAndPathSelection()
        # Should return only critical if path info is missing
        assert strategy.select_bins(base_context) == [2]

    def test_select_bins_with_path(self, base_context):
        # Setup path info: tour might go 0 -> 2 -> 4 -> 0
        # If Bin 2 is critical, and Bin 4 is on path and fits.
        base_context.distance_matrix = np.array([
            [0, 10, 10, 10, 10, 10],
            [10, 0, 10, 10, 10, 10],
            [10, 10, 0, 10, 10, 10],
            [10, 10, 10, 0, 10, 10],
            [10, 10, 10, 10, 0, 10],
            [10, 10, 10, 10, 10, 0]
        ])
        # Bin 2 is index 2 in tour logic (0 is depot)
        # Let's say path 0->2 goes through bin 1.
        base_context.paths_between_states = [
            [[], [], [1], [], [], []], # From 0
            [[], [], [], [], [], []],
            [[0], [], [], [], [], []], # To 0
            # ... rest
        ]
        # Make paths_between_states larger for safety
        n = 6
        base_context.paths_between_states = [[[] for _ in range(n)] for _ in range(n)]
        base_context.paths_between_states[0][2] = [1]
        base_context.paths_between_states[2][0] = []

        base_context.vehicle_capacity = 200.0 # Plenty of space
        strategy = LastMinuteAndPathSelection()
        must_go = strategy.select_bins(base_context)
        # Critical is 2. Path 0->2 contains 1.
        assert 2 in must_go
        assert 1 in must_go

class TestMeansAndStdDevSelection:
    def test_select_bins_no_data(self, base_context):
        strategy = MeansAndStdDevSelection()
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

        strategy = MeansAndStdDevSelection()
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
