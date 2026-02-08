"""
Widget for targeting file system entries.
"""

import os

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)

from ....styles.globals import START_RED_STYLE


class TargetingWidget(QGroupBox):
    """
    Widget for selecting target paths and filename patterns.
    """

    def __init__(self, parent=None):
        """
        Initialize the targeting widget.

        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setLayout(QFormLayout())
        self.layout().setContentsMargins(10, 10, 10, 10)

        # --target_entry (str)
        self.target_entry_input = QLineEdit()
        self.target_entry_input.setPlaceholderText("e.g., /path/to/directory or /path/to/single_file.json")
        self.layout().addRow(
            "Target Entry Path:",
            self._create_browser_layout(self.target_entry_input, is_dir=True),
        )

        # --output_key (str, default=None)
        self.output_key_input = QLineEdit()
        self.output_key_input.setPlaceholderText("e.g., 'overflows' (or the calculated output key)")
        self.layout().addRow("Output Field Key:", self.output_key_input)

        # --filename_pattern (str, default=None)
        self.layout().addRow(QLabel('<span style="font-weight: 600;">Target Directory Settings</span>'))
        self.filename_pattern_input = QLineEdit()
        self.filename_pattern_input.setPlaceholderText("e.g., 'log_*.json'")
        self.layout().addRow("Glob Filename Pattern:", self.filename_pattern_input)

        # --- Separator and Preview Checkbox ---
        self.layout().addRow(QLabel("<hr>"))

        # --update_preview (action='store_true' -> default False)
        self.preview_check = QPushButton("Preview Update")
        self.preview_check.setCheckable(True)
        self.preview_check.setChecked(False)
        self.preview_check.setStyleSheet(START_RED_STYLE)
        self.layout().addRow("Verify changes before updating:", self.preview_check)

    def _create_browser_layout(self, line_edit, is_dir=False):
        """Helper to create a layout with a browse button."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)

        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_path(line_edit, is_dir))
        layout.addWidget(btn)
        return layout

    def _browse_path(self, line_edit, is_dir):
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd())

        if path:
            line_edit.setText(path)

    def get_params(self):
        """Extracts targeting parameters."""
        params = {
            "target_entry": self.target_entry_input.text().strip(),
            "filename_pattern": self.filename_pattern_input.text().strip() or None,
            "output_key": self.output_key_input.text().strip() or None,
        }

        # If the button is UNCHECKED (False), the user wants to PERFORM the update (update_preview=False).
        if not self.preview_check.isChecked():
            params["update_preview"] = False

        return params
