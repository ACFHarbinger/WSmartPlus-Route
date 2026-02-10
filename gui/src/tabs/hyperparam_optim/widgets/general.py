"""
General settings widget for Hyperparameter Optimization (Method, Range, Epochs, Metric).
"""

from gui.src.constants.hpo import HPO_METHODS, HPO_METRICS
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)


class GeneralSettingsWidget(QWidget):
    """
    Widget for configuring core HPO settings like method, range, and optimization metric.
    """

    def __init__(self):
        """
        Initialize GeneralSettingsWidget with form layout.
        """
        super().__init__()
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addRow(QLabel("<b>General Settings</b>"))

        # --hpo_method
        self.hpo_method_combo = QComboBox()
        self.hpo_method_combo.addItems(HPO_METHODS.keys())  # type: ignore[arg-type]
        self.hpo_method_combo.setCurrentText("")
        layout.addRow(QLabel("Optimization Method:"), self.hpo_method_combo)

        # --hpo_range (nargs='+')
        self.hpo_range_input = QLineEdit("0.0 2.0")
        self.hpo_range_input.setPlaceholderText("Min-Max values (space separated)")
        layout.addRow(QLabel("Hyper-Parameter Range:"), self.hpo_range_input)

        # --hpo_epochs
        self.hpo_epochs_input = QSpinBox(minimum=1, maximum=50, value=7)
        layout.addRow(QLabel("Optimization Epochs:"), self.hpo_epochs_input)

        # --metric
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(HPO_METRICS.keys())  # type: ignore[arg-type]
        self.metric_combo.setCurrentText("Validation Loss")
        layout.addRow(QLabel("Metric to Optimize:"), self.metric_combo)

    def get_params(self):
        """
        Extract general HPO parameters from the UI.

        Returns:
            dict: Dictionary containing hpo_method, hpo_range, hpo_epochs, and metric.
        """
        params = {
            "hpo_method": self.hpo_method_combo.currentText(),
            "hpo_epochs": self.hpo_epochs_input.value(),
            "metric": HPO_METRICS[self.metric_combo.currentText()],
        }

        # --hpo_range (nargs='+')
        hpo_range_text = self.hpo_range_input.text().strip()
        if hpo_range_text:
            try:
                params["hpo_range"] = [float(x) for x in hpo_range_text.split()]
            except ValueError:
                print("Warning: hpo_range must contain space-separated floats. Defaulting to None.")
                params["hpo_range"] = None
        else:
            params["hpo_range"] = None

        return params
