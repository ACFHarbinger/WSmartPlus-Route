"""
Main tab for orchestrating and monitoring RL training processes.
"""

import os

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,  # <-- Added
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...components import ClickableHeaderWidget
from ...constants import BASELINES
from ...styles.globals import START_GREEN_STYLE, START_RED_STYLE
from .rl_base import BaseReinforcementLearningTab


class RLTrainingTab(BaseReinforcementLearningTab):
    """Training parameters for Reinforcement Learning"""

    def __init__(self):
        """
        Initialize the RLTrainingTab and setup training hyperparameters.
        """
        super().__init__()
        self.widgets = {}
        self.init_ui()

    def init_ui(self):
        """Initialize the training orchestration UI layout."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()

        layout = QFormLayout()

        # N epochs
        self.widgets["n_epochs"] = QSpinBox()
        self.widgets["n_epochs"].setRange(1, 1000)
        self.widgets["n_epochs"].setValue(25)
        layout.addRow(QLabel("Number of Epochs:"), self.widgets["n_epochs"])

        # Epoch start
        self.widgets["epoch_start"] = QSpinBox()
        self.widgets["epoch_start"].setRange(0, 1000)
        self.widgets["epoch_start"].setValue(0)
        layout.addRow(QLabel("Epoch Start:"), self.widgets["epoch_start"])

        # Learning Rate model
        self.widgets["lr_model"] = QDoubleSpinBox()
        self.widgets["lr_model"].setDecimals(6)
        self.widgets["lr_model"].setRange(0, 1)
        self.widgets["lr_model"].setSingleStep(0.0001)
        self.widgets["lr_model"].setValue(0.0001)
        layout.addRow(QLabel("Learning Rate (Model):"), self.widgets["lr_model"])

        # Learning Rate critic
        self.widgets["lr_critic_value"] = QDoubleSpinBox()
        self.widgets["lr_critic_value"].setDecimals(6)
        self.widgets["lr_critic_value"].setRange(0, 1)
        self.widgets["lr_critic_value"].setSingleStep(0.0001)
        self.widgets["lr_critic_value"].setValue(0.0001)
        layout.addRow(QLabel("Learning Rate (Critic):"), self.widgets["lr_critic_value"])

        # Seed
        self.widgets["seed"] = QSpinBox()
        self.widgets["seed"].setRange(0, 999999)
        self.widgets["seed"].setValue(42)
        layout.addRow(QLabel("Random Seed:"), self.widgets["seed"])

        # Maximum grad norm
        self.widgets["max_grad_norm"] = QDoubleSpinBox()
        self.widgets["max_grad_norm"].setRange(0, 10)
        self.widgets["max_grad_norm"].setValue(1.0)
        layout.addRow(QLabel("Maximum Gradient Norm:"), self.widgets["max_grad_norm"])

        # Exp beta
        self.widgets["exp_beta"] = QDoubleSpinBox()
        self.widgets["exp_beta"].setRange(0, 1)
        self.widgets["exp_beta"].setSingleStep(0.1)
        self.widgets["exp_beta"].setValue(0.8)
        layout.addRow(QLabel("Exponential Beta:"), self.widgets["exp_beta"])

        # Baseline
        self.widgets["baseline"] = QComboBox()
        self.widgets["baseline"].addItem("")
        self.widgets["baseline"].addItems(BASELINES)
        layout.addRow(QLabel("Baseline:"), self.widgets["baseline"])

        # Accumulation steps
        self.widgets["accumulation_steps"] = QSpinBox()
        self.widgets["accumulation_steps"].setRange(1, 100)
        self.widgets["accumulation_steps"].setValue(1)
        layout.addRow(QLabel("Accumulation Steps:"), self.widgets["accumulation_steps"])

        # --- Load Model and Optimizer (Custom Header) ---
        # 1. Create a container widget for the header using the custom clickable class
        self.load_model_optim_header_widget = ClickableHeaderWidget(self._toggle_load_model_optim)
        self.load_model_optim_header_widget.setStyleSheet("QWidget { border: none; padding: 0; margin-top: 5px; }")

        lmo_header_layout = QHBoxLayout(self.load_model_optim_header_widget)
        lmo_header_layout.setContentsMargins(0, 0, 0, 0)
        lmo_header_layout.setSpacing(5)

        # 2. The main text (Standard QLabel)
        self.load_model_optim_label = QLabel("Load Model and Optimizer")

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.load_model_optim_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        # Apply the initial (collapsed) styling to the QLabel
        self.load_model_optim_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3. The clickable toggle button (only the +/- sign)
        self.load_model_optim_toggle_button = QPushButton("+")
        self.load_model_optim_toggle_button.setFlat(True)
        self.load_model_optim_toggle_button.setFixedSize(QSize(20, 20))
        self.load_model_optim_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.load_model_optim_toggle_button.clicked.connect(self._toggle_load_model_optim)

        # 4. Add components to the header layout
        lmo_header_layout.addWidget(self.load_model_optim_label)
        lmo_header_layout.addStretch()
        lmo_header_layout.addWidget(self.load_model_optim_toggle_button)

        # 5. Add the header widget to the main layout, making it span the row
        layout.addRow(self.load_model_optim_header_widget)

        # 6. Create a container for the collapsible content
        self.load_model_optim_container = QWidget()
        load_model_optim_layout = QFormLayout(self.load_model_optim_container)
        load_model_optim_layout.setContentsMargins(0, 0, 0, 0)

        # 7. Add widgets to the container's layout
        # --load_path
        self.widgets["load_path"] = QLineEdit()
        self.widgets["load_path"].setPlaceholderText("Path to load model parameters and optimizer state from")
        load_model_optim_layout.addRow("Load Path:", self._create_browser_layout(self.widgets["load_path"]))

        # --resume
        self.widgets["resume"] = QLineEdit()
        self.widgets["resume"].setPlaceholderText("Resume from previous checkpoint file")
        load_model_optim_layout.addRow("Resume From:", self._create_browser_layout(self.widgets["resume"]))

        # 8. Add the content container to the main layout
        layout.addWidget(self.load_model_optim_container)

        # 9. Initialize state: hidden
        self.is_load_visible = False
        self.load_model_optim_container.hide()

        # Checkboxes
        self.widgets["eval_only"] = QPushButton("Evaluation Only")
        self.widgets["eval_only"].setCheckable(True)
        self.widgets["eval_only"].setChecked(False)
        self.widgets["eval_only"].setStyleSheet(START_RED_STYLE)
        layout.addRow(QLabel("Options:"), self.widgets["eval_only"])
        self.widgets["no_cuda"] = QPushButton("Use CUDA")
        self.widgets["no_cuda"].setCheckable(True)
        self.widgets["no_cuda"].setChecked(False)
        self.widgets["no_cuda"].setStyleSheet(START_GREEN_STYLE)
        layout.addRow("", self.widgets["no_cuda"])

        self.widgets["enable_scaler"] = QPushButton("Enable CUDA Scaler")
        self.widgets["enable_scaler"].setCheckable(True)
        self.widgets["enable_scaler"].setStyleSheet(START_RED_STYLE)
        layout.addRow("", self.widgets["enable_scaler"])

        self.widgets["checkpoint_encoder"] = QPushButton("Checkpoint Encoder (Decrease Memory Usage)")
        self.widgets["checkpoint_encoder"].setCheckable(True)
        self.widgets["checkpoint_encoder"].setStyleSheet(START_RED_STYLE)
        layout.addRow("", self.widgets["checkpoint_encoder"])

        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def _create_browser_layout(self, line_edit, is_dir=False):
        """
        Helper to create a layout with a line edit and a browse button.

        Args:
            line_edit (QLineEdit): The line edit to populate.
            is_dir (bool): Whether to browse for a directory or a file.

        Returns:
            QHBoxLayout: Layout containing the widgets.
        """
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)

        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_path(line_edit, is_dir))
        layout.addWidget(btn)
        return layout

    def _browse_path(self, line_edit, is_dir):
        """
        Open a file/directory dialog and set the selected path.

        Args:
            line_edit (QLineEdit): Target line edit.
            is_dir (bool): Whether to browse for a directory.
        """
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd())

        if path:
            line_edit.setText(path)

    def _toggle_load_model_optim(self):
        """Toggles the visibility of the Load Model and Optimizer input fields and updates the +/- sign."""
        if self.is_load_visible:
            self.load_model_optim_container.hide()
            self.load_model_optim_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.load_model_optim_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.load_model_optim_container.show()
            self.load_model_optim_toggle_button.setText("-")

            # Remove the border from the QLabel when expanded.
            self.load_model_optim_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_load_visible = not self.is_load_visible

    def get_params(self):
        """
        Collect all training parameters from the UI components.

        Returns:
            dict: Dictionary of training hyperparameters and session flags.
        """
        params = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QSpinBox):
                val = widget.value()
                if key == "epoch_start" and val != 0 or key != "epoch_start":
                    params[key] = val
            elif isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if text:
                    params[key] = text
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                if text:
                    params[key] = text
            elif isinstance(widget, QCheckBox):
                if widget.isChecked():
                    params[key] = True

        # This part handles boolean flags that might not be QCheckBoxes.
        params["eval_only"] = self.widgets["eval_only"].isChecked()
        params["no_cuda"] = self.widgets["no_cuda"].isChecked()
        params["enable_scaler"] = self.widgets["enable_scaler"].isChecked()
        params["checkpoint_encoder"] = self.widgets["checkpoint_encoder"].isChecked()
        return params
