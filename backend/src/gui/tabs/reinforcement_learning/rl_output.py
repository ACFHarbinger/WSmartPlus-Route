from PySide6.QtWidgets import (
    QSpinBox, QLabel,
    QComboBox, QLineEdit,
    QPushButton, QFormLayout,
)
from backend.src.gui.app_definitions import WB_MODES
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
        layout.addRow(QLabel("Log Directory:"), self.widgets['log_dir'])
        
        # Run name
        self.widgets['run_name'] = QLineEdit()
        layout.addRow(QLabel("Run Name:"), self.widgets['run_name'])
        
        # Output dir
        self.widgets['output_dir'] = QLineEdit("model_weights")
        layout.addRow(QLabel("Output Directory:"), self.widgets['output_dir'])
        
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
        
        # Checkboxes´
        start_green_style = """
            QPushButton:checked {
                background-color: #8B0000;
                color: white;
            }
            QPushButton {
                background-color: #06402B;
                color: white;
            }
        """
        self.widgets['no_tensorboard'] = QPushButton("TensorBoard Logger")
        self.widgets['no_tensorboard'].setChecked(False)
        self.widgets['no_tensorboard'].setCheckable(True)
        self.widgets['no_tensorboard'].setStyleSheet(start_green_style)
        layout.addRow(QLabel("Options:"), self.widgets['no_tensorboard'])

        self.widgets['no_progress_bar'] = QPushButton("Progress Bar")
        self.widgets['no_progress_bar'].setChecked(False)
        self.widgets['no_progress_bar'].setCheckable(True)
        self.widgets['no_progress_bar'].setStyleSheet(start_green_style)
        layout.addRow("", self.widgets['no_progress_bar'])
        
        self.setLayout(layout)
        
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
            elif isinstance(widget, QPushButton):
                if widget.isChecked():
                    params[key] = True
        return params
