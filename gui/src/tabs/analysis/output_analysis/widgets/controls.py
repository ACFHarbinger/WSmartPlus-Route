"""
Control widgets for Output Analysis dashboard.
"""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class OutputControlsWidget(QWidget):
    """
    Horizontal control bar for loading and configuring output analysis plots.
    """

    def __init__(self, parent=None):
        """
        Initialize the control widget and create UI elements (buttons, comboboxes).
        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.load_btn = QPushButton("Load Output File(s) (JSON/JSONL/TBL)")
        layout.addWidget(self.load_btn)

        self.dist_combo = QComboBox()
        self.dist_combo.setToolTip("Filter plots by data distribution (emp, gammaX, etc.)")
        layout.addWidget(QLabel("Distribution:"))
        layout.addWidget(self.dist_combo)

        self.x_key_combo = QComboBox()
        self.x_key_combo.setPlaceholderText("X-Axis")
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.x_key_combo)

        self.y_key_combo = QComboBox()
        self.y_key_combo.setPlaceholderText("Y-Axis")
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.y_key_combo)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"])
        layout.addWidget(QLabel("Type:"))
        layout.addWidget(self.chart_type_combo)

        self.pareto_check = QCheckBox("Pareto Front")
        self.pareto_check.setToolTip("Highlight non-dominated solutions (Min X, Min Y)")
        layout.addWidget(self.pareto_check)

        self.plot_btn = QPushButton("Plot Chart")
        self.plot_btn.setEnabled(False)
        layout.addWidget(self.plot_btn)

        layout.addStretch()
