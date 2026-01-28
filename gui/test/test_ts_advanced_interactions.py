import pytest

from unittest.mock import patch
from gui.src.constants import VERTEX_METHODS
from gui.src.tabs.test_simulator.ts_advanced import TestSimAdvancedTab


class TestSimAdvancedInteractions:
    @pytest.fixture
    def adv_tab(self, qapp):
        return TestSimAdvancedTab()

    def test_key_files_toggle(self, adv_tab):
        """Test Key/License Files section toggle."""
        assert not adv_tab.is_key_license_files_visible
        assert adv_tab.key_license_files_container.isHidden()

        adv_tab._toggle_key_license_files()
        assert adv_tab.is_key_license_files_visible
        assert not adv_tab.key_license_files_container.isHidden()

    def test_get_params_defaults(self, adv_tab):
        """Test default parameter retrieval."""
        params = adv_tab.get_params()

        assert params["cpu_cores"] == 1
        assert params["no_progress_bar"] is True
        assert params["server_run"] is False
        assert params["resume"] is False

        # Default Vertex Method: "Min-Max Normalization" -> "mmn" (assuming constant)
        # Using the constant map to verify
        default_v_text = "Min-Max Normalization"
        if default_v_text in VERTEX_METHODS:
            assert params["vertex_method"] == VERTEX_METHODS[default_v_text]

    def test_get_params_custom(self, adv_tab):
        """Test valid custom inputs."""
        with patch("multiprocessing.cpu_count", return_value=4):
            adv_tab.cpu_cores_input.setMaximum(16)
            adv_tab.cpu_cores_input.setValue(4)
            adv_tab.env_file_input.setText("production.env")
            adv_tab.edge_threshold_input.setValue(0.75)

            # Toggle flags
            adv_tab.server_run_check.setChecked(True)
            adv_tab.no_progress_check.setChecked(False)

            # Keys (Need to toggle visible? Code doesn't check visibility for params, just text existence)
            adv_tab.hexlic_file_input.setText("/path/to/hexaly.lic")

            params = adv_tab.get_params()

            assert params["cpu_cores"] == 4
            assert params["env_file"] == "production.env"
            assert params["edge_threshold"] == 0.75
            assert params["server_run"] is True
            assert params["no_progress_bar"] is False
            assert params["hexlic_file"] == "/path/to/hexaly.lic"
