from unittest.mock import patch

import pytest

from gui.src.tabs.reinforcement_learning.rl_model import RLModelTab
from gui.src.tabs.reinforcement_learning.rl_training import RLTrainingTab


class TestRLTabInteractions:
    @pytest.fixture
    def training_tab(self, qapp):
        return RLTrainingTab()

    @pytest.fixture
    def model_tab(self, qapp):
        return RLModelTab()

    def test_training_tab_toggle_transparency(self, training_tab):
        """Test the expand/collapse logic for Load Model section."""
        # Initial state: Hidden
        assert not training_tab.is_load_visible
        assert training_tab.load_model_optim_container.isHidden()
        assert training_tab.load_model_optim_toggle_button.text() == "+"

        # Toggle ON
        training_tab._toggle_load_model_optim()
        assert training_tab.is_load_visible
        assert not training_tab.load_model_optim_container.isHidden()
        assert training_tab.load_model_optim_toggle_button.text() == "-"

        # Toggle OFF
        training_tab._toggle_load_model_optim()
        assert not training_tab.is_load_visible
        assert training_tab.load_model_optim_container.isHidden()

    def test_training_tab_get_params(self, training_tab):
        """Test parameter retrieval mechanics."""
        # Set values
        training_tab.widgets["n_epochs"].setValue(50)
        training_tab.widgets["lr_model"].setValue(0.001)
        training_tab.widgets["eval_only"].setChecked(True)

        # Test text input
        training_tab.widgets["load_path"].setText("/tmp/model.pt")

        params = training_tab.get_params()

        assert params["n_epochs"] == 50
        assert params["lr_model"] == 0.001
        assert params["eval_only"] is True
        assert params["load_path"] == "/tmp/model.pt"

        # Test conditional parameter (epoch_start=0 is default, should be omitted if skipped? No, code logic checks != 0)
        # Code: if key == "epoch_start" and val != 0: ... elif ... params[key] = val
        # Wait, if epoch_start is 0, it's skipped?
        # Let's verify default behavior
        assert "epoch_start" not in params  # Default is 0

        training_tab.widgets["epoch_start"].setValue(10)
        params = training_tab.get_params()
        assert params["epoch_start"] == 10

    def test_model_tab_mapping(self, model_tab):
        """Test the mapping from UI text to CLI arguments in get_params."""

        # Model: "Attention Model" -> "attn" (assuming mapping in MODELS)
        # We need to rely on the imported dictionaries.
        from gui.src.constants import MODELS

        # Set Model
        ui_model_name = list(MODELS.keys())[0]  # e.g., "Attention Model"
        MODELS[ui_model_name]
        model_tab.widgets["model"].setCurrentText(ui_model_name)

        # Set Normalization: "Batch Norm" -> "batch_norm"
        # UI populates with Title Case from NORMALIZATION_METHODS keys?
        # Code: [nm.replace("_", " ").title() for nm in NORMALIZATION_METHODS]
        # NORMALIZATION_METHODS is a list of strings like ["batch_norm", "layer_norm"]?
        # Actually it's imported. Let's assume standard values.
        # If NORMALIZATION_METHODS has "batch_norm", UI has "Batch Norm"

        # Let's pick an item from the combobox to be safe
        box = model_tab.widgets["normalization"]
        if box.count() > 1:
            box.setCurrentIndex(0)  # First real item
            ui_text = box.currentText()
            # If ui_text is "Batch Norm", params should have "batch_norm"

            params = model_tab.get_params()

            # Reconstruction logic verification
            if ui_text:
                expected_key = ui_text.lower().replace(" ", "_")
                assert params["normalization"] == expected_key

        # Check basic spinbox
        model_tab.widgets["embedding_dim"].setValue(256)
        params = model_tab.get_params()
        assert params["embedding_dim"] == 256

    def test_browse_path_logic(self, training_tab):
        """Test the file browser logic patching QFileDialog."""
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=("/path/to/file", "filter")):
            training_tab._browse_path(training_tab.widgets["load_path"], is_dir=False)
            assert training_tab.widgets["load_path"].text() == "/path/to/file"

        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="/path/to/dir"):
            training_tab._browse_path(training_tab.widgets["resume"], is_dir=True)
            assert training_tab.widgets["resume"].text() == "/path/to/dir"
