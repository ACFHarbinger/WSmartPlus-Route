import pytest

from unittest.mock import patch
from gui.src.tabs.hyperparam_optim import HyperParamOptimParserTab


class TestHPOInteractions:
    @pytest.fixture
    def hpo_tab(self, qapp):
        tab = HyperParamOptimParserTab()
        yield tab
        tab.close()
        tab.deleteLater()
        qapp.processEvents()

    def test_timeout_toggle(self, hpo_tab):
        """Test Timeout section toggle."""
        # Access timeout widget via algorithms widget
        timeout_widget = hpo_tab.algorithms_widget.timeout_widget
        assert not timeout_widget.is_visible
        assert timeout_widget.content_container.isHidden()

        timeout_widget._toggle()
        assert timeout_widget.is_visible
        assert not timeout_widget.content_container.isHidden()

        timeout_widget._toggle()
        assert not timeout_widget.is_visible

    def test_get_params_basic(self, hpo_tab):
        """Test basic parameters."""
        with patch("multiprocessing.cpu_count", return_value=4):
            # cpu_cores is in ray_tune_widget
            hpo_tab.ray_tune_widget.cpu_cores_input.setMaximum(16)
            # hpo_epochs is in general_widget
            hpo_tab.general_widget.hpo_epochs_input.setValue(15)
            hpo_tab.ray_tune_widget.cpu_cores_input.setValue(4)

            params = hpo_tab.get_params()

            assert params["hpo_epochs"] == 15
            assert params["cpu_cores"] == 4
            assert params["train_best"] is True  # Default

    def test_method_mapping(self, hpo_tab):
        """Test HPO Method mapping."""
        # "Bayesian Optimization (BO)" -> "bo"
        ui_text = "Bayesian Optimization (BO)"
        # hpo_method_combo is in general_widget
        hpo_tab.general_widget.hpo_method_combo.setCurrentText(ui_text)

        params = hpo_tab.get_params()
        # Dictionary keys are strings
        # HPO_METHODS keys are the UI strings
        # HPO_METHODS = {"Bayesian ...": "bo", ...}
        # Code: params["hpo_method"] = self.hpo_method_combo.currentText()
        # Wait, looked at code:
        # "hpo_method": self.hpo_method_combo.currentText()
        # It returns the raw text?
        # Let's double check implementation of get_params in hyperparam_optim.py.
        # Line 262: "hpo_method": self.hpo_method_combo.currentText(),
        # It does NOT look it up in dictionary?
        # The script likely handles mapping or the combobox uses keys?
        # In init: self.hpo_method_combo.addItems(HPO_METHODS.keys())
        # So it returns the UI text.
        # OK, let's assert it returns the UI text then.

        assert params["hpo_method"] == ui_text

    def test_range_parsing(self, hpo_tab):
        """Test float list parsing."""
        # hpo_range_input in general_widget
        hpo_tab.general_widget.hpo_range_input.setText("1.0 5.0")
        # grid_input in algorithms_widget
        hpo_tab.algorithms_widget.grid_input.setText("0 1 2")

        params = hpo_tab.get_params()

        assert params["hpo_range"] == [1.0, 5.0]
        assert params["grid"] == [0.0, 1.0, 2.0]

    def test_timeout_parsing(self, hpo_tab):
        """Test valid and invalid timeout inputs."""
        # timeout_input is inside algorithms_widget.timeout_widget.input
        hpo_tab.algorithms_widget.timeout_widget.input.setText("60")
        params = hpo_tab.get_params()
        assert params["timeout"] == 60

        hpo_tab.algorithms_widget.timeout_widget.input.setText("invalid")
        params = hpo_tab.get_params()
        assert params["timeout"] is None
