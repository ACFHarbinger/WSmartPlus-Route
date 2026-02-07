from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .widgets.inplace import InplaceUpdateWidget
from .widgets.statistics import StatisticsUpdateWidget
from .widgets.targeting import TargetingWidget


class FileSystemUpdateTab(QWidget):
    """
    GUI tab for the 'update' command, focused on modifying contents
    or attributes of file system entries. The contents are wrapped in a
    QScrollArea for scrollability.
    """

    def __init__(self):
        super().__init__()
        # 1. Create a container widget for all content
        scrollable_content = QWidget()
        # 2. Create the layout that will hold all the settings and be scrollable
        content_layout = QVBoxLayout(scrollable_content)

        # Add the title
        content_layout.addWidget(QLabel("<h2>Update File System Entries</h2>"))

        # --- Targeting Widget ---
        self.targeting_widget = TargetingWidget()
        content_layout.addWidget(self.targeting_widget)

        # --- Update Logic Group (update_operation, update_value, update_preview) ---
        logic_group = QGroupBox()
        logic_layout = QFormLayout(logic_group)
        logic_layout.addRow(QLabel('<span style="font-weight: 600;">Update Logic Settings</span>'))

        # --- Inplace Update Widget ---
        self.inplace_widget = InplaceUpdateWidget()
        logic_layout.addRow(self.inplace_widget)

        # --- Statistics Update Widget ---
        self.statistics_widget = StatisticsUpdateWidget()
        logic_layout.addRow(self.statistics_widget)

        content_layout.addWidget(logic_group)
        content_layout.addStretch()  # Push content to the top

        # 4. Create the QScrollArea and set the content widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scrollable_content)

        # 5. Set the final layout of the tab (self)
        tab_layout = QVBoxLayout(self)
        tab_layout.addWidget(scroll_area)

    def get_params(self):
        """Extracts settings into a dictionary mimicking argparse output."""
        params = {}

        # 1. Targeting params
        params.update(self.targeting_widget.get_params())

        # 2. Inplace params
        params.update(self.inplace_widget.get_params())

        # 3. Statistics params
        params.update(self.statistics_widget.get_params())

        # Final cleanup for empty strings that should be None
        for key in ["target_entry", "filename_pattern", "output_key"]:
            if key in params and not params[key]:
                params[key] = None

        return params
