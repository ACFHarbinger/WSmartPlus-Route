from unittest.mock import patch

import pytest

from gui.src.tabs.file_system.fs_update import FileSystemUpdateTab


class TestFileSystemInteractions:
    @pytest.fixture
    def fs_tab(self, qapp):
        tab = FileSystemUpdateTab()
        yield tab
        tab.close()
        tab.deleteLater()
        qapp.processEvents()

    def test_inplace_update_toggle(self, fs_tab):
        """Test Inplace Update header toggle."""
        assert not fs_tab.inplace_widget.is_inplace_update_visible
        assert fs_tab.inplace_widget.inplace_update_container.isHidden()

        fs_tab.inplace_widget._toggle_inplace_update()
        assert fs_tab.inplace_widget.is_inplace_update_visible
        assert not fs_tab.inplace_widget.inplace_update_container.isHidden()

        fs_tab.inplace_widget._toggle_inplace_update()
        assert not fs_tab.inplace_widget.is_inplace_update_visible

    def test_stats_update_toggle(self, fs_tab):
        """Test Stats Update header toggle."""
        assert not fs_tab.statistics_widget.is_stats_update_visible

        fs_tab.statistics_widget._toggle_stats_update()
        assert fs_tab.statistics_widget.is_stats_update_visible

        fs_tab.statistics_widget._toggle_stats_update()
        assert not fs_tab.statistics_widget.is_stats_update_visible

    def test_get_params_basic(self, fs_tab):
        """Test basic parameter retrieval."""
        fs_tab.targeting_widget.target_entry_input.setText("/tmp/data.json")
        fs_tab.targeting_widget.output_key_input.setText("profit")

        params = fs_tab.get_params()

        assert params["target_entry"] == "/tmp/data.json"
        assert params["output_key"] == "profit"
        assert (
            params.get("update_preview") is False
        )  # Default checked=False -> update_preview=False (logic inverted in code?)

        # Code logic: if not self.preview_check.isChecked(): params["update_preview"] = False
        # Button defaults to False (Unchecked). So update_preview = False.
        # Wait, usually "Preview" means "Dry run".
        # If checked (True), then update_preview is True (implied by absence of False override? No, arg default is likely True/False in CLI).
        # Let's check logic:
        # if not checked: params["update_preview"] = False.
        # So key is present only if False? Or present as False?
        assert params["update_preview"] is False

        # Set preview to True
        fs_tab.targeting_widget.preview_check.setChecked(True)
        params = fs_tab.get_params()
        # If checked, the 'if not checked' block is skipped.
        # Depending on how the command uses it (argparse store_true?), if it's missing it might be default.
        # But get_params returns a dict. If key is missing, CLI might use default.
        # But test should verify what get_params returns.
        assert "update_preview" not in params

    def test_get_params_complex(self, fs_tab):
        """Test complex inplace/stats parameters."""
        # Enable sections
        fs_tab.inplace_widget._toggle_inplace_update()
        fs_tab.statistics_widget._toggle_stats_update()

        # Inplace
        fs_tab.inplace_widget.update_operation_combo.setCurrentText("Set value (=)")  # Maps to '='
        fs_tab.inplace_widget.update_value_input.setText("123.45")
        fs_tab.inplace_widget.input_key_1_input.setText("k1")

        # Stats
        fs_tab.statistics_widget.update_function_combo.setCurrentText("Mean")  # Maps to 'mean'
        fs_tab.statistics_widget.output_filename_input.setText("stats.json")

        params = fs_tab.get_params()

        assert params["update_operation"] == "="
        assert params["update_value"] == 123.45
        assert "k1" in params["input_keys"]

        assert params["stats_function"] == "mean"
        assert params["output_filename"] == "stats.json"

    def test_browse_button(self, fs_tab):
        """Test browse logic."""
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="/tmp/dir"):
            fs_tab.targeting_widget._browse_path(fs_tab.targeting_widget.target_entry_input, is_dir=True)
            assert fs_tab.targeting_widget.target_entry_input.text() == "/tmp/dir"
