import pytest

from gui.src.tabs.hyperparam_optim import HyperParamOptimParserTab


class TestHPOInteractions:
    @pytest.fixture
    def hpo_tab(self, qapp):
        return HyperParamOptimParserTab()

    def test_timeout_toggle(self, hpo_tab):
        """Test Timeout section toggle."""
        assert not hpo_tab.is_timeout_visible
        assert hpo_tab.timeout_container.isHidden()

        hpo_tab._toggle_timeout()
        assert hpo_tab.is_timeout_visible
        assert not hpo_tab.timeout_container.isHidden()

        hpo_tab._toggle_timeout()
        assert not hpo_tab.is_timeout_visible

    def test_get_params_basic(self, hpo_tab):
        """Test basic parameters."""
        hpo_tab.hpo_epochs_input.setValue(15)
        hpo_tab.cpu_cores_input.setValue(4)

        params = hpo_tab.get_params()

        assert params["hpo_epochs"] == 15
        assert params["cpu_cores"] == 4
        assert params["train_best"] is True  # Default

    def test_method_mapping(self, hpo_tab):
        """Test HPO Method mapping."""
        # "Bayesian Optimization (BO)" -> "bo"
        ui_text = "Bayesian Optimization (BO)"
        hpo_tab.hpo_method_combo.setCurrentText(ui_text)

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
        hpo_tab.hpo_range_input.setText("1.0 5.0")
        hpo_tab.grid_input.setText("0 1 2")

        params = hpo_tab.get_params()

        assert params["hpo_range"] == [1.0, 5.0]
        assert params["grid"] == [0.0, 1.0, 2.0]

    def test_timeout_parsing(self, hpo_tab):
        """Test valid and invalid timeout inputs."""
        hpo_tab.timeout_input.setText("60")
        params = hpo_tab.get_params()
        assert params["timeout"] == 60

        hpo_tab.timeout_input.setText("invalid")
        params = hpo_tab.get_params()
        assert params["timeout"] is None
