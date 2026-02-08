"""
Widget for selecting and browsing scripts to execute.
"""

import sys

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....components import ClickableHeaderWidget
from ....constants.scripts import SCRIPTS
from ....styles.globals import (
    SCRIPT_HEADER_STYLE,
    SECONDARY_BUTTON_STYLE,
)


class ScriptSelectionWidget(QWidget):
    """
    Widget for selecting a script to run.
    Handles the grid layout of script buttons and the Linux-only toggle.
    """

    scriptSelected = Signal(str)
    selectionCleared = Signal()

    def __init__(self):
        """
        Initialize the selection widget, generating buttons for each available script.
        """
        super().__init__()

        self.SCRIPTS = SCRIPTS
        self.selected_script = None
        self.script_buttons = {}

        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Button Group for exclusivity
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # Determine which scripts to display
        all_script_names = list(self.SCRIPTS.keys())
        if sys.platform.startswith("linux"):
            scripts_to_display = all_script_names
            linux_only_scripts = []
        else:
            scripts_to_display = all_script_names[:-2]
            linux_only_scripts = all_script_names[-2:]

        # --- Dynamic 2x2 Layout Generation ---
        i = 0
        while i < len(scripts_to_display):
            row_layout = QHBoxLayout()
            for j in range(2):
                if i + j < len(scripts_to_display):
                    script_name = scripts_to_display[i + j]
                    btn = self._create_script_button(script_name)
                    row_layout.addWidget(btn)
                else:
                    row_layout.addStretch(1)
                    break
            layout.addLayout(row_layout)
            i += 2

        # Clear Selection Button
        clear_btn_layout = QHBoxLayout()
        self.btn_clear_selection = QPushButton("Clear Selection")
        self.btn_clear_selection.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.btn_clear_selection.clicked.connect(self.clear_selection)
        clear_btn_layout.addStretch()
        clear_btn_layout.addWidget(self.btn_clear_selection)
        layout.addLayout(clear_btn_layout)

        # Linux Only Scripts Section (Hidden on non-Linux)
        if not sys.platform.startswith("linux"):
            self._setup_linux_only_section(layout, linux_only_scripts)

    def _create_script_button(self, script_name):
        """Helper method to create script buttons"""
        display_name = self.SCRIPTS[script_name]
        btn = QPushButton(display_name)
        btn.setCheckable(True)
        btn.setObjectName("toggleStyleButton")

        btn.clicked.connect(lambda checked, s=script_name: self._on_script_clicked(s, checked))
        self.button_group.addButton(btn)
        self.script_buttons[script_name] = btn
        return btn

    def _on_script_clicked(self, script_name, checked):
        """
        Internal handler for script button toggles.

        Args:
            script_name (str): Key of the clicked script.
            checked (bool): Toggle state.
        """
        if checked:
            self.selected_script = script_name
            self.scriptSelected.emit(script_name)
        else:
            self.selected_script = None
            self.selectionCleared.emit()

    def clear_selection(self):
        """Clear the current script selection"""
        current_button = self.button_group.checkedButton()
        if current_button:
            self.button_group.setExclusive(False)
            current_button.setChecked(False)
            self.button_group.setExclusive(True)

        self.selected_script = None
        self.selectionCleared.emit()

    def _setup_linux_only_section(self, parent_layout, linux_only_scripts):
        """Setup the collapsible Linux-only scripts section"""
        self.linux_only_scripts_header_widget = ClickableHeaderWidget(self._toggle_linux_only_scripts)
        self.linux_only_scripts_header_widget.setStyleSheet("QWidget { border: none; padding: 0; margin-top: 10px; }")

        los_header_layout = QHBoxLayout(self.linux_only_scripts_header_widget)
        los_header_layout.setContentsMargins(0, 0, 0, 0)
        los_header_layout.setSpacing(5)

        self.linux_only_scripts_label = QLabel("Linux Only Scripts (Disabled)")
        self.linux_only_scripts_label.setStyleSheet(SCRIPT_HEADER_STYLE)

        self.linux_only_scripts_toggle_button = QPushButton("+")
        self.linux_only_scripts_toggle_button.setFlat(True)
        self.linux_only_scripts_toggle_button.setFixedSize(QSize(20, 20))
        self.linux_only_scripts_toggle_button.setStyleSheet(
            """
            QPushButton {
                font-weight: 600;
                padding: 0;
                border: none;
                background: transparent;
                color: #7F8C8D; /* Muted text */
            }
            """
        )
        self.linux_only_scripts_toggle_button.clicked.connect(self._toggle_linux_only_scripts)

        los_header_layout.addWidget(self.linux_only_scripts_label)
        los_header_layout.addStretch()
        los_header_layout.addWidget(self.linux_only_scripts_toggle_button)

        parent_layout.addWidget(self.linux_only_scripts_header_widget)

        self.linux_only_scripts_container = QWidget()
        linux_only_scripts_layout = QVBoxLayout(self.linux_only_scripts_container)
        linux_only_scripts_layout.setContentsMargins(0, 5, 0, 0)

        self.linux_buttons_container = QWidget()
        row_slurm_layout = QHBoxLayout(self.linux_buttons_container)
        row_slurm_layout.setContentsMargins(0, 0, 0, 0)

        self.linux_buttons = []

        for script_name in linux_only_scripts:
            btn = self._create_script_button(script_name)

            # New "warning" disabled style
            current_style = btn.styleSheet() if btn.styleSheet() else ""
            btn.setStyleSheet(
                current_style
                + """
                QPushButton:disabled {
                    color: #B87A00; /* Dark Yellow/Orange */
                    background-color: #FFF9E6; /* Light Yellow */
                    border: 1px solid #FFEBC6;
                }
                """
            )
            btn.setEnabled(False)

            self.linux_buttons.append(btn)
            row_slurm_layout.addWidget(btn)

        if len(linux_only_scripts) == 1:
            row_slurm_layout.addStretch(1)

        linux_only_scripts_layout.addWidget(self.linux_buttons_container)
        parent_layout.addWidget(self.linux_only_scripts_container)
        self.is_area_visible = False
        self.linux_only_scripts_container.hide()

    def _toggle_linux_only_scripts(self):
        """Toggles the visibility of the Linux Only Scripts input fields."""
        if self.is_area_visible:
            self.linux_only_scripts_container.hide()
            self.linux_only_scripts_toggle_button.setText("+")
            self.linux_only_scripts_label.setText("Linux Only Scripts (Disabled)")
        else:
            self.linux_only_scripts_container.show()
            self.linux_only_scripts_toggle_button.setText("–")  # Use en-dash
            self.linux_only_scripts_label.setText("Linux Only Scripts (Disabled)")
        self.is_area_visible = not self.is_area_visible
