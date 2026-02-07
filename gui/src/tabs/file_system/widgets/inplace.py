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
from ....constants import OPERATION_MAP


class InplaceUpdateWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 2.1 Create a container widget for the header using the custom clickable class
        self.inplace_update_header_widget = ClickableHeaderWidget(self._toggle_inplace_update)
        self.inplace_update_header_widget.setStyleSheet("QWidget { border: none; padding: 0; margin-top: 5px; }")

        iu_header_layout = QHBoxLayout(self.inplace_update_header_widget)
        iu_header_layout.setContentsMargins(0, 0, 0, 0)
        iu_header_layout.setSpacing(5)

        # 2.2 The main text (Standard QLabel)
        self.inplace_update_label = QLabel('<span style="font-weight: 600;">Inplace Update Parameters</span>')

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.inplace_update_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        # Apply the initial (collapsed) styling to the QLabel
        self.inplace_update_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 2.3 The clickable toggle button (only the +/- sign)
        self.inplace_update_toggle_button = QPushButton("+")
        self.inplace_update_toggle_button.setFlat(True)
        self.inplace_update_toggle_button.setFixedSize(QSize(20, 20))
        self.inplace_update_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.inplace_update_toggle_button.clicked.connect(self._toggle_inplace_update)

        # 2.4 Add components to the header layout
        iu_header_layout.addWidget(self.inplace_update_label)
        iu_header_layout.addStretch()
        iu_header_layout.addWidget(self.inplace_update_toggle_button)

        # 2.5 Add the header widget to the logic layout
        self.layout.addRow(self.inplace_update_header_widget)

        # 2.6 Create a container for the collapsible content
        self.inplace_update_container = QWidget()
        inplace_update_layout = QFormLayout(self.inplace_update_container)
        inplace_update_layout.setContentsMargins(0, 0, 0, 0)

        # 2.7 Add widgets to the container's layout
        # --update_operation (default=None)
        self.update_operation_combo = QComboBox()
        self.update_operation_combo.addItems(OPERATION_MAP.keys())
        self.update_operation_combo.setCurrentIndex(-1)  # Start with no selection
        self.update_operation_combo.setPlaceholderText("Select Operation")
        inplace_update_layout.addRow("Update Operation:", self.update_operation_combo)

        # --update_value (float, default=0.0)
        self.update_value_input = QLineEdit("0.0")
        self.update_value_input.setPlaceholderText("Enter a float or string value")
        inplace_update_layout.addRow("Update Value (for single-input):", self.update_value_input)

        # --- Input Keys for Two-Input Calculation ---
        # Container for the two horizontal input boxes
        input_keys_h_layout = QHBoxLayout()

        self.input_key_1_input = QLineEdit()
        self.input_key_1_input.setPlaceholderText("Key 1 (e.g., 'total_miles')")
        input_keys_h_layout.addWidget(self.input_key_1_input)

        self.input_key_2_input = QLineEdit()
        self.input_key_2_input.setPlaceholderText("Key 2 (e.g., 'cost_per_mile')")
        input_keys_h_layout.addWidget(self.input_key_2_input)

        # Add the horizontal layout to the form layout with a label
        inplace_update_layout.addRow("Input Keys (for two-input calculation):", input_keys_h_layout)

        # 2.8 Add the content container to the logic layout
        self.layout.addRow(self.inplace_update_container)

        # 2.9 Initialize state: hidden
        self.is_inplace_update_visible = False
        self.inplace_update_container.hide()

    def _toggle_inplace_update(self):
        """Toggles the visibility of the Inplace Update input fields and updates the +/- sign."""
        if self.is_inplace_update_visible:
            self.inplace_update_container.hide()
            self.inplace_update_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.inplace_update_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.inplace_update_container.show()
            self.inplace_update_toggle_button.setText("-")

            # Remove the border from the QLabel when expanded.
            self.inplace_update_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_inplace_update_visible = not self.is_inplace_update_visible

    def get_params(self):
        """Extracts parameters relative to inplace updates."""
        params = {}
        if self.is_inplace_update_visible and self.update_operation_combo.currentText().strip():
            # 3.1a Get update operation (must be one of the choices)
            selected_function = self.update_operation_combo.currentText()
            params["update_operation"] = OPERATION_MAP.get(selected_function, "")

            # 3.2a Get update value (Try to cast to float as requested by type=float)
            update_value_str = self.update_value_input.text().strip()
            try:
                # Try to convert to float, otherwise keep as string/default
                params["update_value"] = float(update_value_str)
            except ValueError:
                # If it's not a valid float, keep it as the string, as it might be used
                # for 'replace_string' or similar functions.
                params["update_value"] = update_value_str if update_value_str else 0.0

            # 3.3a Handle --input_keys (nargs='*', default=(None, None))
            key1 = self.input_key_1_input.text().strip() or False
            key2 = self.input_key_2_input.text().strip() or False
            params["input_keys"] = "{}{}".format(key1 if key1 else "", f" {key2}" if key2 else "")
        return params
