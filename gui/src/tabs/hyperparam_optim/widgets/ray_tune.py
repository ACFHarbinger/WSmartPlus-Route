"""
Settings widget for Ray Tune framework configuration (CPUs, Verbosity, etc.).
"""

import multiprocessing as mp

from gui.src.styles.globals import START_RED_STYLE
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)


class RayTuneSettingsWidget(QWidget):
    """
    Widget for configuring Ray Tune execution parameters like CPU cores and verbosity.
    """

    def __init__(self):
        """
        Initialize RayTuneSettingsWidget with form layout.
        """
        super().__init__()
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addRow(QLabel("<b>Ray Tune Framework Settings</b>"))

        # --cpu_cores
        system_cpu_cores = mp.cpu_count()
        self.cpu_cores_input = QSpinBox(minimum=1, maximum=system_cpu_cores, value=1)
        layout.addRow(QLabel("CPU Cores:"), self.cpu_cores_input)

        # --verbose
        self.verbose_input = QSpinBox(minimum=0, maximum=3, value=2)
        layout.addRow(QLabel("Verbose Level (0-3):"), self.verbose_input)

        # --train_best
        self.train_best_check = QPushButton("Train final model with best hyper-parameters")
        self.train_best_check.setCheckable(True)
        self.train_best_check.setChecked(True)
        self.train_best_check.setStyleSheet(START_RED_STYLE)
        layout.addRow(QLabel("Train Best Model:"), self.train_best_check)

        # --local_mode
        self.local_mode_check = QPushButton("Run Ray in Local Mode")
        self.local_mode_check.setCheckable(True)
        self.local_mode_check.setChecked(False)
        self.local_mode_check.setStyleSheet(START_RED_STYLE)
        layout.addRow(QLabel("Local Mode:"), self.local_mode_check)

        # --num_samples
        self.num_samples_input = QSpinBox(minimum=1, maximum=1000, value=20)
        layout.addRow(QLabel("Number of Samples:"), self.num_samples_input)

    def get_params(self):
        """
        Extract Ray Tune specific parameters from the UI.

        Returns:
            dict: Dictionary containing cpu_cores, verbose, train_best, local_mode, and num_samples.
        """
        return {
            "cpu_cores": self.cpu_cores_input.value(),
            "verbose": self.verbose_input.value(),
            "train_best": self.train_best_check.isChecked(),
            "local_mode": self.local_mode_check.isChecked(),
            "num_samples": self.num_samples_input.value(),
        }
