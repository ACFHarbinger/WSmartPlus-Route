from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ....components import ClickableHeaderWidget
from ....constants import FUNCTION_MAP


class StatisticsUpdateWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 3.1 Create a container widget for the header using the custom clickable class
        self.stats_update_header_widget = ClickableHeaderWidget(self._toggle_stats_update)
        self.stats_update_header_widget.setStyleSheet("QWidget { border: none; padding: 0; margin-top: 5px; }")

        su_header_layout = QHBoxLayout(self.stats_update_header_widget)
        su_header_layout.setContentsMargins(0, 0, 0, 0)
        su_header_layout.setSpacing(5)

        # 3.2 The main text (Standard QLabel)
        self.stats_update_label = QLabel('<span style="font-weight: 600;">Statistics Update Parameters</span>')

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.stats_update_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        # Apply the initial (collapsed) styling to the QLabel
        self.stats_update_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3.3 The clickable toggle button (only the +/- sign)
        self.stats_update_toggle_button = QPushButton("+")
        self.stats_update_toggle_button.setFlat(True)
        self.stats_update_toggle_button.setFixedSize(QSize(20, 20))
        self.stats_update_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.stats_update_toggle_button.clicked.connect(self._toggle_stats_update)

        # 3.4 Add components to the header layout
        su_header_layout.addWidget(self.stats_update_label)
        su_header_layout.addStretch()
        su_header_layout.addWidget(self.stats_update_toggle_button)

        # 3.5 Add the header widget to the logic layout
        self.layout.addRow(self.stats_update_header_widget)

        # 3.6 Create a container for the collapsible content
        self.stats_update_container = QWidget()
        stats_update_layout = QFormLayout(self.stats_update_container)
        stats_update_layout.setContentsMargins(0, 0, 0, 0)

        # 3.7 Add widgets to the container's layout
        # --stats_function (default=None)
        self.update_function_combo = QComboBox()
        self.update_function_combo.addItems(FUNCTION_MAP.keys())
        self.update_function_combo.addItem("")
        self.update_function_combo.setCurrentIndex(-1)  # Start with no selection
        self.update_function_combo.setPlaceholderText("Select Statistics Function")
        stats_update_layout.addRow("Update Function:", self.update_function_combo)

        # --output_filename (str, default=None)
        self.output_filename_input = QLineEdit()
        self.output_filename_input.setPlaceholderText("Name of output file (e.g., log_mean.json)")
        stats_update_layout.addRow("Output Filename:", self.output_filename_input)

        # 3.8 Add the content container to the logic layout
        self.layout.addRow(self.stats_update_container)

        # 3.9 Initialize state: hidden
        self.is_stats_update_visible = False
        self.stats_update_container.hide()

    def _toggle_stats_update(self):
        """Toggles the visibility of the Statistics Update input fields and updates the +/- sign."""
        if self.is_stats_update_visible:
            self.stats_update_container.hide()
            self.stats_update_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.stats_update_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.stats_update_container.show()
            self.stats_update_toggle_button.setText("-")

            # Remove the border from the QLabel when expanded.
            self.stats_update_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_stats_update_visible = not self.is_stats_update_visible

    def get_params(self):
        """Extracts parameters relative to statistics calculation."""
        params = {}
        if self.is_stats_update_visible and self.update_function_combo.currentText().strip():
            # 4.1 Get update function (must be one of the choices)
            selected_function = self.update_function_combo.currentText()
            params["stats_function"] = FUNCTION_MAP.get(selected_function, "")

            # 4.2 Get name of output file
            params["output_filename"] = self.output_filename_input.text().strip()
        return params
