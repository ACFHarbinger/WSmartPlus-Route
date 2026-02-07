import multiprocessing as mp

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ....constants.scripts import SCRIPTS
from ....styles.globals import (
    SCRIPT_HEADER_STYLE,
    SECTION_HEADER_STYLE,
)


class ScriptParametersWidget(QWidget):
    """
    Widget for configuring script execution parameters.
    Handles common parameters (verbose, cores) and script-specific inputs.
    """

    def __init__(self):
        super().__init__()

        self.SCRIPTS = SCRIPTS
        self.param_widgets = {}

        # Main Layout
        layout = QFormLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # --- Common Parameters Section ---
        params_header = QLabel("Script Parameters")
        params_header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addRow(params_header)

        # Selected Script Display
        self.selected_script_label = QLabel("No script selected")
        self.selected_script_label.setStyleSheet(
            """
            font-weight: 600;
            color: #007AFF; /* Primary Accent */
            background-color: #E6F2FF; /* Light blue */
            border-radius: 4px;
            padding: 5px 8px;
        """
        )
        layout.addRow("Selected Script:", self.selected_script_label)

        # Common toggles
        self.verbose_checkbox = QPushButton("Verbose Mode")
        self.verbose_checkbox.setCheckable(True)
        self.verbose_checkbox.setObjectName("toggleStyleButton")
        layout.addRow("Verbose Output:", self.verbose_checkbox)

        self.cores_input = QSpinBox(value=22, minimum=1, maximum=mp.cpu_count())
        layout.addRow("CPU Cores:", self.cores_input)

        layout.addRow(self._create_separator())

        # --- Script-Specific Parameters Section ---
        specific_params_header = QLabel("Script-Specific Parameters")
        specific_params_header.setStyleSheet(SCRIPT_HEADER_STYLE)
        layout.addRow(specific_params_header)

        self.script_params_container = QWidget()
        self.script_params_layout = QFormLayout(self.script_params_container)
        self.script_params_layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(self.script_params_container)

        self.script_params_container.setVisible(False)

        self._init_script_parameters()

    def _create_separator(self):
        """Creates a modern, thin horizontal separator."""
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #DDE3E8;")  # BORDER_COLOR
        return separator

    def _init_script_parameters(self):
        """Initialize parameter widgets for each script type"""
        # Environment setup parameters
        setup_env_params = QWidget(self.script_params_container)
        setup_env_layout = QFormLayout(setup_env_params)
        setup_env_layout.setContentsMargins(0, 0, 0, 0)

        self.env_manager_input = QComboBox()
        self.env_manager_input.addItems(["uv", "conda", "venv"])
        setup_env_layout.addRow("Package Manager:", self.env_manager_input)

        self.param_widgets["setup_env"] = setup_env_params
        self.script_params_layout.addWidget(setup_env_params)

        # Default placeholder
        self.default_params_label = QLabel("No specific parameters for this script.")
        self.default_params_label.setStyleSheet("color: #7F8C8D;")  # Muted text
        self.script_params_layout.addWidget(self.default_params_label)

        setup_env_params.hide()
        self.default_params_label.hide()

    def update_selection(self, script_name):
        """Update the UI to reflect selected script"""
        display_name = self.SCRIPTS[script_name]
        self.selected_script_label.setText(display_name)
        self._show_specific_params(script_name)

    def clear_selection(self):
        """Reset the UI to no selection state"""
        self.selected_script_label.setText("No script selected")
        self._hide_specific_params()

    def _show_specific_params(self, script_name):
        """Show parameters specific to the selected script."""
        self.script_params_container.setVisible(True)

        for widget in self.param_widgets.values():
            widget.hide()
        self.default_params_label.hide()

        if script_name in self.param_widgets:
            self.param_widgets[script_name].show()
        else:
            self.default_params_label.setText(f"No specific parameters for {self.SCRIPTS[script_name]}")
            self.default_params_label.show()

    def _hide_specific_params(self):
        """Hide the script parameters section."""
        self.script_params_container.setVisible(False)
        for widget in self.param_widgets.values():
            widget.hide()
        self.default_params_label.hide()

    def get_params(self, selected_script):
        """Get all parameters for the selected script"""
        if not selected_script:
            return {}

        base_params = {
            "script": selected_script,
            "verbose": self.verbose_checkbox.isChecked(),
            "cores": self.cores_input.value(),
        }

        if selected_script == "setup_env":
            base_params.update(
                {
                    "manager": self.env_manager_input.currentText(),
                }
            )

        return base_params
