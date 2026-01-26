import pytest

from gui.src.constants import SIMULATOR_TEST_POLICIES
from gui.src.tabs.test_simulator.ts_policy_parameters import TestSimPolicyParamsTab
from gui.src.tabs.test_simulator.ts_settings import TestSimSettingsTab


class TestSimTabInteractions:
    @pytest.fixture
    def settings_tab(self, qapp):
        return TestSimSettingsTab()

    @pytest.fixture
    def policy_params_tab(self, qapp):
        return TestSimPolicyParamsTab()

    def test_settings_tab_policy_selection(self, settings_tab):
        """Test policy selection mechanics (buttons + logic)."""
        # Initial state: Empty
        assert len(settings_tab.selected_policies) == 0

        # Test individual toggle
        policy_name = list(SIMULATOR_TEST_POLICIES.keys())[0]
        settings_tab.toggle_policy(policy_name, True)
        assert policy_name in settings_tab.selected_policies

        settings_tab.toggle_policy(policy_name, False)
        assert policy_name not in settings_tab.selected_policies

        # Test Select All
        settings_tab.select_all_policies()
        assert len(settings_tab.selected_policies) == len(SIMULATOR_TEST_POLICIES)

        # Test Deselect All
        settings_tab.deselect_all_policies()
        assert len(settings_tab.selected_policies) == 0

    def test_settings_tab_get_params(self, settings_tab):
        """Test parameter retrieval mechanics."""
        # Set some values
        settings_tab.problem_input.setCurrentText("WCVRP")
        settings_tab.size_input.setValue(100)
        settings_tab.days_input.setValue(5)

        # Select one policy
        policy_key = list(SIMULATOR_TEST_POLICIES.keys())[0]  # e.g. "Attention Model"
        settings_tab.toggle_policy(policy_key, True)

        params = settings_tab.get_params()

        assert params["problem"] == "wcvrp"
        assert params["size"] == 100
        assert params["days"] == 5

        # Verify policies string
        expected_pol_val = SIMULATOR_TEST_POLICIES[policy_key]
        assert params["policies"] == expected_pol_val

        # Default Distribution: "Gamma 1" -> "gamma1"
        # Check if default is gamma1
        if settings_tab.data_dist_input.currentText() == "Gamma 1":
            assert params["data_distribution"] == "gamma1"

    def test_policy_params_tab_get_params(self, policy_params_tab):
        """Test policy parameters form logic."""
        policy_params_tab.decode_type_combo.setCurrentText("Sampling")
        policy_params_tab.temperature_input.setValue(2.5)

        # Boolean flags
        policy_params_tab.lookahead_config_a.setChecked(True)
        policy_params_tab.lookahead_config_b.setChecked(True)
        policy_params_tab.run_tsp_check.setChecked(True)

        # Text inputs
        policy_params_tab.gurobi_param_input.setText("0.5 0.9")

        params = policy_params_tab.get_params()

        assert params["decode_type"] == "sampling"
        assert params["temperature"] == 2.5
        assert params["run_tsp"] is True

        # Lookahead configs should be "a b"
        assert "a" in params["lookahead_configs"]
        assert "b" in params["lookahead_configs"]

        assert params["gurobi_param"] == "0.5 0.9"
