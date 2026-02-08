"""
Live dashboard and summary UI components for simulation results.
"""

from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LiveDashboardTab(QWidget):
    """
    Encapsulates the live dashboard tab logic and widgets.
    """

    selectionChanged = Signal()
    viewRouteRequested = Signal(str, int)  # key, day

    def __init__(self, policy_names: List[str], parent=None):
        """
        Initialize the live dashboard.

        Args:
            policy_names: List of all policy names to display.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.policy_names = policy_names
        self.setup_ui()

    def setup_ui(self):
        """Build the dashboard layout."""
        layout = QVBoxLayout(self)

        # Control Panel
        controls = QHBoxLayout()

        self.policy_combo = QComboBox()
        self.policy_combo.addItems(self.policy_names)
        self.policy_combo.currentIndexChanged.connect(self.selectionChanged.emit)

        self.sample_combo = QComboBox()
        self.sample_combo.currentIndexChanged.connect(self.selectionChanged.emit)

        self.metric_combo = QComboBox()
        self.metric_combo.currentIndexChanged.connect(self.selectionChanged.emit)

        controls.addWidget(QLabel("Policy:"))
        controls.addWidget(self.policy_combo)
        controls.addWidget(QLabel("Sample:"))
        controls.addWidget(self.sample_combo)
        controls.addWidget(QLabel("Metric:"))
        controls.addWidget(self.metric_combo)
        controls.addStretch()

        layout.addLayout(controls)

        # Chart Areas (Placeholders for where Matplotlib layouts go)
        self.line_chart_layout = QVBoxLayout()
        self.bar_chart_layout = QVBoxLayout()

        main_charts = QHBoxLayout()
        main_charts.addLayout(self.line_chart_layout, 2)
        main_charts.addLayout(self.bar_chart_layout, 1)

        layout.addLayout(main_charts)

        # Bottom Area: Day selection and Route viewing
        bottom_controls = QHBoxLayout()
        self.day_combo = QComboBox()
        self.view_route_btn = QPushButton("View Route on Map")
        self.view_route_btn.setEnabled(False)

        bottom_controls.addWidget(QLabel("Select Day for Details:"))
        bottom_controls.addWidget(self.day_combo)
        bottom_controls.addWidget(self.view_route_btn)
        bottom_controls.addStretch()

        layout.addLayout(bottom_controls)

    def update_samples(self, samples: List[str]):
        """Update the sample list while keeping selection if possible."""
        current = self.sample_combo.currentText()
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        self.sample_combo.addItems(sorted(samples))
        idx = self.sample_combo.findText(current)
        if idx >= 0:
            self.sample_combo.setCurrentIndex(idx)
        self.sample_combo.blockSignals(False)

    def update_metrics(self, metrics: List[str]):
        """Update the metric list."""
        current = self.metric_combo.currentText()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        self.metric_combo.addItems(sorted(metrics))
        idx = self.metric_combo.findText(current)
        if idx >= 0:
            self.metric_combo.setCurrentIndex(idx)
        self.metric_combo.blockSignals(False)
