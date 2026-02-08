"""
UI Components for RunScriptsTab.
"""

from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...styles.globals import SECTION_HEADER_STYLE
from .builder import get_command
from .widgets.parameters import ScriptParametersWidget
from .widgets.selection import ScriptSelectionWidget


class RunScriptsTab(QWidget):
    """
    Main tab for configuring and running system scripts (training, testing, HPO, etc.).
    """

    def __init__(self):
        """
        Initialize the RunScriptsTab and setup the selection and parameter widgets.
        """
        super().__init__()

        # 1. Create the content widget
        self.content_widget = QWidget()

        # 2. Use QVBoxLayout for the content (simpler stacking of widgets)
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # --- Script Selection Section ---
        script_header = QLabel("Script Selection")
        script_header.setStyleSheet(SECTION_HEADER_STYLE)
        content_layout.addWidget(script_header)

        self.selection_widget = ScriptSelectionWidget()
        content_layout.addWidget(self.selection_widget)

        content_layout.addWidget(self._create_separator())

        # --- Script Parameters Section ---
        self.parameters_widget = ScriptParametersWidget()
        content_layout.addWidget(self.parameters_widget)

        # --- Signal Connections ---
        self.selection_widget.scriptSelected.connect(self._on_script_selected)
        self.selection_widget.selectionCleared.connect(self._on_selection_cleared)

        content_layout.addStretch()

        # --- Make Tab Scrollable ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.content_widget)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        self.content_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

    def _create_separator(self):
        """Creates a modern, thin horizontal separator."""
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #DDE3E8;")  # BORDER_COLOR
        return separator

    def _on_script_selected(self, script_name):
        """Update parameter widget when a script is selected."""
        self.parameters_widget.update_selection(script_name)

    def _on_selection_cleared(self):
        """Clear parameter widget when selection is cleared."""
        self.parameters_widget.clear_selection()

    def get_command(self):
        """
        Generate the final shell command based on user selections and parameters.

        Returns:
            str: The formatted command string, or None if no script is selected.
        """
        selected_script = self.selection_widget.selected_script
        if not selected_script:
            return None

        params = self.parameters_widget.get_params(selected_script)
        return get_command(params)
