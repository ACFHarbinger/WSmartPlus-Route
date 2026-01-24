from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
)

from .rl_base import BaseReinforcementLearningTab


class RLCostsTab(BaseReinforcementLearningTab):
    """Cost function weights"""

    def __init__(self):
        super().__init__()
        self.widgets = {}
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        # Waste weight
        self.widgets["w_waste"] = QDoubleSpinBox()
        self.widgets["w_waste"].setRange(-10, 10)
        self.widgets["w_waste"].setSingleStep(0.1)
        self.widgets["w_waste"].setSpecialValueText("Not set")
        self.widgets["w_waste"].setValue(0)
        layout.addRow(QLabel("Waste Weight:"), self.widgets["w_waste"])

        # Length weight
        self.widgets["w_length"] = QDoubleSpinBox()
        self.widgets["w_length"].setRange(-10, 10)
        self.widgets["w_length"].setSingleStep(0.1)
        self.widgets["w_length"].setSpecialValueText("Not set")
        self.widgets["w_length"].setValue(0)
        layout.addRow(QLabel("Length Weight:"), self.widgets["w_length"])

        # Overflows weight
        self.widgets["w_overflows"] = QDoubleSpinBox()
        self.widgets["w_overflows"].setRange(-10, 10)
        self.widgets["w_overflows"].setSingleStep(0.1)
        self.widgets["w_overflows"].setSpecialValueText("Not set")
        self.widgets["w_overflows"].setValue(0)
        layout.addRow(QLabel("Overflows Weight:"), self.widgets["w_overflows"])

        # Lost weight
        self.widgets["w_lost"] = QDoubleSpinBox()
        self.widgets["w_lost"].setRange(-10, 10)
        self.widgets["w_lost"].setSingleStep(0.1)
        self.widgets["w_lost"].setSpecialValueText("Not set")
        self.widgets["w_lost"].setValue(0)
        layout.addRow(QLabel("Lost Weight:"), self.widgets["w_lost"])

        # Penalty weight
        self.widgets["w_penalty"] = QDoubleSpinBox()
        self.widgets["w_penalty"].setRange(-10, 10)
        self.widgets["w_penalty"].setSingleStep(0.1)
        self.widgets["w_penalty"].setSpecialValueText("Not set")
        self.widgets["w_penalty"].setValue(0)
        layout.addRow(QLabel("Penalty Weight:"), self.widgets["w_penalty"])

        # Prize weight
        self.widgets["w_prize"] = QDoubleSpinBox()
        self.widgets["w_prize"].setRange(-10, 10)
        self.widgets["w_prize"].setSingleStep(0.1)
        self.widgets["w_prize"].setSpecialValueText("Not set")
        self.widgets["w_prize"].setValue(0)
        layout.addRow(QLabel("Prize Weight:"), self.widgets["w_prize"])

        self.setLayout(layout)

    def get_params(self):
        params = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                val = widget.value()
                if val != 0:
                    params[key] = val
        return params
