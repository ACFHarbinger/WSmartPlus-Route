import pytest

from gui.src.tabs.meta_rl_train import MetaRLTrainParserTab


class TestMetaRLInteractions:
    @pytest.fixture
    def meta_tab(self, qapp):
        tab = MetaRLTrainParserTab()
        yield tab
        tab.close()
        tab.deleteLater()
        qapp.processEvents()

    def test_get_params_basics(self, meta_tab):
        """Test basic inputs and simple spinboxes."""
        meta_tab.mrl_history_input.setValue(20)
        meta_tab.mrl_lr_input.setValue(0.005)

        params = meta_tab.get_params()

        assert params["mrl_history"] == 20
        assert params["mrl_lr"] == 0.005

    def test_mapping_logic(self, meta_tab):
        """Test ComboBox to CLI argument mapping."""
        # Clean current text to ensure we test setting it
        # Pick "Multi-Objective Reinforcement Learning (MORL)" -> "morl"
        ui_text = "Multi-Objective Reinforcement Learning (MORL)"
        expected_arg = "morl"

        # Ensure item exists
        if meta_tab.mrl_method_combo.findText(ui_text) >= 0:
            meta_tab.mrl_method_combo.setCurrentText(ui_text)
            params = meta_tab.get_params()
            assert params["mrl_method"] == expected_arg

        # Test CB Exploration: "Thompson Sampling" -> "thompson_sampling"
        meta_tab.cb_exploration_method_combo.setCurrentText("Thompson Sampling")
        params = meta_tab.get_params()
        assert params["cb_exploration_method"] == "thompson_sampling"

    def test_list_parsing(self, meta_tab):
        """Test parsing of space-separated strings into lists."""
        # Floats
        meta_tab.mrl_range_input.setText("0.1 0.5 1.0")
        # Strings
        meta_tab.cb_context_features_input.setText("feat1 feat2")

        params = meta_tab.get_params()

        assert params["mrl_range"] == [0.1, 0.5, 1.0]
        assert params["cb_context_features"] == ["feat1", "feat2"]

    def test_invalid_list_input(self, meta_tab):
        """Test invalid float list input handles gracefully."""
        meta_tab.mrl_range_input.setText("0.1 invalid 1.0")
        params = meta_tab.get_params()
        # Should not be present (skipped)
        assert "mrl_range" not in params
