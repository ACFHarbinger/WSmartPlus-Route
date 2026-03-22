"""
Hyperparameter optimization configuration tab.
"""

from PySide6.QtWidgets import (
    QFormLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .widgets.algorithms import AlgorithmSettingsWidget
from .widgets.general import GeneralSettingsWidget
from .widgets.ray_tune import RayTuneSettingsWidget


class HyperParamOptimParserTab(QWidget):
    """
    Tab for configuring Hyperparameter Optimization (HPO) arguments.
    """

    def __init__(self):
        """
        Initialize the HyperParamOptimParserTab and setup configuration widgets.
        """
        super().__init__()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        self.form_layout = QFormLayout(content)

        # 1. General Settings
        self.general_widget = GeneralSettingsWidget()
        self.form_layout.addRow(self.general_widget)

        # 2. Ray Tune Settings
        self.ray_tune_widget = RayTuneSettingsWidget()
        self.form_layout.addRow(self.ray_tune_widget)

        # 3. Algorithms Settings (BO, DEA, HBO, GS, DEHBO, RS)
        self.algorithms_widget = AlgorithmSettingsWidget()
        self.form_layout.addRow(self.algorithms_widget)

        QVBoxLayout(self).addWidget(scroll_area)
        scroll_area.setWidget(content)

    def get_params(self):
        """
        Collect all hyperparameter optimization parameters from sub-widgets.

        Returns:
            dict: Dictionary containing general, ray tune, and algorithm-specific parameters.
        """
        params = {}
        params.update(self.general_widget.get_params())
        params.update(self.ray_tune_widget.get_params())
        params.update(self.algorithms_widget.get_params())
        return params
