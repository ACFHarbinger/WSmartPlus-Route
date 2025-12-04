import os
from PySide6.QtWidgets import (
    QSpinBox, QLabel,
    QComboBox, QLineEdit,
    QPushButton, QFormLayout,
    QHBoxLayout, QFileDialog # <-- Added
)
from ...app_definitions import WB_MODES
from ...styles import START_GREEN_STYLE
from .rl_base import BaseReinforcementLearningTab


class RLOutputTab(BaseReinforcementLearningTab):
    """Reinforcement Learning Logging parameters"""
    def __init__(self):
        super().__init__()
        self.widgets = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout()
        
        # Log step
        self.widgets['log_step'] = QSpinBox()
        self.widgets['log_step'].setRange(1, 10000)
        self.widgets['log_step'].setValue(50)
        layout.addRow(QLabel("Log Step:"), self.widgets['log_step'])
        
        # Log dir
        self.widgets['log_dir'] = QLineEdit("logs")
        layout.addRow(QLabel("Log Directory:"), self._create_browser_layout(self.widgets['log_dir'], is_dir=True))
        
        # Run name
        self.widgets['run_name'] = QLineEdit()
        layout.addRow(QLabel("Run Name:"), self.widgets['run_name'])
        
        # Output dir
        self.widgets['output_dir'] = QLineEdit("model_weights")
        layout.addRow(QLabel("Output Directory:"), self._create_browser_layout(self.widgets['output_dir'], is_dir=True))
        
        # Checkpoint epochs
        self.widgets['checkpoint_epochs'] = QSpinBox()
        self.widgets['checkpoint_epochs'].setRange(0, 100)
        self.widgets['checkpoint_epochs'].setValue(1)
        layout.addRow(QLabel("Checkpoint Epochs:"), self.widgets['checkpoint_epochs'])
        
        # Wandb mode
        self.widgets['wandb_mode'] = QComboBox()
        self.widgets['wandb_mode'].addItems(WB_MODES)
        self.widgets['wandb_mode'].addItem('')
        layout.addRow(QLabel("Weight and Biases Mode:"), self.widgets['wandb_mode'])
        
        # Checkboxes
        self.widgets['no_tensorboard'] = QPushButton("TensorBoard Logger")
        self.widgets['no_tensorboard'].setCheckable(True)
        self.widgets['no_tensorboard'].setChecked(False)
        self.widgets['no_tensorboard'].setStyleSheet(START_GREEN_STYLE)
        layout.addRow(QLabel("Options:"), self.widgets['no_tensorboard'])

        self.widgets['no_progress_bar'] = QPushButton("Progress Bar")
        self.widgets['no_progress_bar'].setCheckable(True)
        self.widgets['no_progress_bar'].setChecked(False)
        self.widgets['no_progress_bar'].setStyleSheet(START_GREEN_STYLE)
        layout.addRow("", self.widgets['no_progress_bar'])
        
        self.setLayout(layout)

    def _create_browser_layout(self, line_edit, is_dir=False):
        """Helper to create a layout with a browse button."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_path(line_edit, is_dir))
        layout.addWidget(btn)
        return layout

    def _browse_path(self, line_edit, is_dir):
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd())
        
        if path:
            line_edit.setText(path)
        
    def get_params(self):
        params = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if text:
                    params[key] = text
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                if text:
                    params[key] = text
        
        params['no_tensorboard'] = self.widgets['no_tensorboard'].isChecked()
        params['no_progress_bar'] = self.widgets['no_progress_bar'].isChecked()
        return params