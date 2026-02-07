"""
Summary statistics and heatmap UI components for simulation results.
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


class SummaryStatisticsTab(QWidget):
    """
    Encapsulates the summary statistics tab logic and widgets.
    """

    selectionChanged = Signal()
    redrawRequested = Signal()

    def __init__(self, policy_names: List[str], parent=None):
        super().__init__(parent)
        self.policy_names = policy_names
        self.setup_ui()

    def setup_ui(self):
        """Build the summary area layout."""
        layout = QVBoxLayout(self)

        # Header with Comparison controls
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Simulation Summary Comparison</b>"))
        header.addStretch()

        self.redraw_btn = QPushButton("Refresh Comparison")
        self.redraw_btn.clicked.connect(self.redrawRequested.emit)
        header.addWidget(self.redraw_btn)

        layout.addLayout(header)

        # Large Comparison Plot Area
        self.comparison_chart_layout = QVBoxLayout()
        layout.addLayout(self.comparison_chart_layout, 2)

        # Heatmap / Individual Analysis Section
        analysis_header = QHBoxLayout()
        analysis_header.addWidget(QLabel("<b>Detailed Heatmap Analysis</b>"))
        analysis_header.addStretch()
        layout.addLayout(analysis_header)

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

        # Heatmap Area
        self.heatmap_chart_layout = QVBoxLayout()
        layout.addLayout(self.heatmap_chart_layout, 1)

    def update_samples(self, samples: List[str]):
        """Update the sample list."""
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
