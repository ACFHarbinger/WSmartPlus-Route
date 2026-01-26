from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.src.constants import LR_SCHEDULERS, OPTIMIZERS

from .rl_base import BaseReinforcementLearningTab


class RLOptimizerTab(BaseReinforcementLearningTab):
    """Optimizer and scheduler parameters for Reinforcement Learning"""

    def __init__(self):
        super().__init__()
        self.widgets = {}
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()

        layout = QFormLayout()

        # Optimizer
        self.widgets["optimizer"] = QComboBox()
        self.widgets["optimizer"].addItems(OPTIMIZERS.keys())
        self.widgets["optimizer"].setCurrentText("Root Mean Square Propagation (RMSProp)")
        layout.addRow(QLabel("Optimizer:"), self.widgets["optimizer"])

        # Learning Rate Scheduler
        self.widgets["lr_scheduler"] = QComboBox()
        self.widgets["lr_scheduler"].addItems(LR_SCHEDULERS.keys())
        self.widgets["lr_scheduler"].setCurrentText("Lambda Learning Rate")
        layout.addRow(QLabel("Learning Rate Scheduler:"), self.widgets["lr_scheduler"])

        # Learning Rate decay
        self.widgets["lr_decay"] = QDoubleSpinBox()
        self.widgets["lr_decay"].setRange(0, 2)
        self.widgets["lr_decay"].setSingleStep(0.1)
        self.widgets["lr_decay"].setValue(1.0)
        layout.addRow(QLabel("Learning Rate Decay:"), self.widgets["lr_decay"])

        # Learning Rate minimum value
        self.widgets["lr_min_value"] = QDoubleSpinBox()
        self.widgets["lr_min_value"].setDecimals(6)
        self.widgets["lr_min_value"].setRange(0, 1)
        self.widgets["lr_min_value"].setValue(0.0)
        layout.addRow(QLabel("Learning Rate Minimum Value:"), self.widgets["lr_min_value"])

        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def get_params(self):
        params = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                val = widget.value()
                # Only add if value is non-zero, as per original logic
                if val != 0.0:
                    params[key] = val
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                if not text:
                    continue  # Skip if empty text

                # 1. Determine the dictionary based on the widget's KEY ('optimizer' or 'lr_scheduler')
                lookup_dict = None
                if key == "optimizer":
                    lookup_dict = OPTIMIZERS
                elif key == "lr_scheduler":
                    lookup_dict = LR_SCHEDULERS

                # 2. Perform the lookup using the selected UI text
                if lookup_dict:
                    # Look up the short CLI value (e.g., 'rmsprop') using the full UI text
                    cli_argument = lookup_dict.get(text, None)
                    if cli_argument:
                        params[key] = cli_argument
                    # If the lookup fails (e.g., for custom entry), fall back to the raw text
                    else:
                        params[key] = text
                else:
                    # If no specific dictionary is matched (shouldn't happen here), use raw text
                    params[key] = text

        return params
