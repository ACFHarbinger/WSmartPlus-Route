import pytest

from gui.src.constants import DATA_DISTRIBUTIONS

# Fix import path: gd_problem is a module in gui.src.tabs.generate_data
from gui.src.tabs.generate_data.gd_problem import GenDataProblemTab


class TestGenDataInteractions:
    @pytest.fixture
    def gd_tab(self, qapp):
        return GenDataProblemTab()

    def test_initial_state(self, gd_tab):
        """Test initial state of GenDataProblemTab."""
        assert gd_tab.problem_combo.currentText() == "All"
        assert gd_tab.graph_sizes_input.text() == "20 50 100"

        # Check that 'All' distribution button is checked by default if present
        if "All" in gd_tab.dist_buttons:
            assert gd_tab.dist_buttons["All"].isChecked()

    def test_select_deselect_all_dists(self, gd_tab):
        """Test Select All and Deselect All buttons."""
        # Deselect all first
        gd_tab.deselect_all_distributions()
        for btn in gd_tab.dist_buttons.values():
            assert not btn.isChecked()

        # Select all
        gd_tab.select_all_distributions()
        for btn in gd_tab.dist_buttons.values():
            assert btn.isChecked()

    def test_get_params(self, gd_tab):
        """Test parameter retrieval logic."""
        gd_tab.problem_combo.setCurrentText("CVRPP")
        gd_tab.graph_sizes_input.setText("10 20")

        # Setup specific distribution selection
        gd_tab.deselect_all_distributions()

        # Pick a couple of distributions to check
        # Assuming DATA_DISTRIBUTIONS keys exist
        keys = list(DATA_DISTRIBUTIONS.keys())
        if len(keys) >= 2:
            dist1 = keys[0]
            dist2 = keys[1]

            gd_tab.dist_buttons[dist1].setChecked(True)
            gd_tab.dist_buttons[dist2].setChecked(True)

            params = gd_tab.get_params()

            assert params["problem"] == "CVRPP"
            assert params["graph_sizes"] == "10 20"

            # Verify data_distributions string contains mapped values
            val1 = DATA_DISTRIBUTIONS[dist1]
            val2 = DATA_DISTRIBUTIONS[dist2]

            # The order depends on iteration order of dist_buttons dict (usually insertion order in Py3.7+)
            # So "val1 val2" or "val2 val1" depending on key order
            actual_dists = params["data_distributions"].split()
            assert val1 in actual_dists
            assert val2 in actual_dists
            assert len(actual_dists) == 2

    def test_get_params_defaults(self, gd_tab):
        """Test default parameters."""
        gd_tab.select_all_distributions()
        params = gd_tab.get_params()

        # All selected
        assert len(params["data_distributions"].split()) == len(DATA_DISTRIBUTIONS)
