from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from gui.src.utils.app_definitions import DECODE_TYPES


class TestSimPolicyParamsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        layout.addRow(QLabel("<b>Attention Model Parameters</b>"))

        # 1. --decode_type
        self.decode_type_combo = QComboBox(currentText="Greedy")
        self.decode_type_combo.addItems([dt.title() for dt in DECODE_TYPES])
        layout.addRow("Decode Type:", self.decode_type_combo)

        # 2. --temperature
        self.temperature_input = QDoubleSpinBox(value=1.0, minimum=0.0, maximum=5.0, singleStep=0.1)
        layout.addRow("Softmax Temperature:", self.temperature_input)

        layout.addRow(QLabel("<b>Policy Parameters</b>"))
        layout.addRow(QLabel('<span style="font-weight: 600;">Space-separated lists</span>'))

        # 3. --pregular_level
        self.pregular_level_input = QLineEdit()
        self.pregular_level_input.setPlaceholderText("e.g., 2 3 6 (for --lvl)")
        layout.addRow("Regular Policy Level:", self.pregular_level_input)

        # 4. --plastminute_cf
        self.plastminute_cf_input = QLineEdit()
        self.plastminute_cf_input.setPlaceholderText("e.g., 50 70 90 (for --cf)")
        layout.addRow("Last Minute CF:", self.plastminute_cf_input)

        # 5. --gurobi_param
        self.gurobi_param_input = QLineEdit()
        self.gurobi_param_input.setPlaceholderText("e.g., 0.42 0.84 (for --gp)")
        layout.addRow("Gurobi VRPP Parameter:", self.gurobi_param_input)

        # 6. --hexaly_param
        self.hexaly_param_input = QLineEdit()
        self.hexaly_param_input.setPlaceholderText("e.g., 0.42 0.84 (for --hp)")
        layout.addRow("Hexaly VRPP Parameter:", self.hexaly_param_input)

        # --- Boolean Flags ---
        layout.addRow(QLabel('<span style="font-weight: 600;">Boolean Flags</span>'))
        boolean_flags_layout = QHBoxLayout()
        self.lookahead_config_a = QCheckBox("Look-Ahead Configuration A")  # 7. lookahead_configs
        self.lookahead_config_b = QCheckBox("Look-Ahead Configuration B")
        self.cache_regular_check = QCheckBox("Deactivate Regular Cache")  # 8. --cache_regular
        self.run_tsp_check = QCheckBox("Run fast_tsp for routing")  # 9. --run_tsp

        # Add widgets to the horizontal layout (using addWidget)
        boolean_flags_layout.addWidget(self.lookahead_config_a)
        boolean_flags_layout.addWidget(self.lookahead_config_b)
        boolean_flags_layout.addWidget(self.cache_regular_check)
        boolean_flags_layout.addWidget(self.run_tsp_check)

        # Add the entire horizontal layout as a single row to the QFormLayout
        layout.addRow(boolean_flags_layout)

    def get_params(self):
        look_ahead_configs = []
        if self.lookahead_config_a.isChecked():
            look_ahead_configs.append("a")
        if self.lookahead_config_b.isChecked():
            look_ahead_configs.append("b")
        params = {
            "decode_type": self.decode_type_combo.currentText().lower(),
            "temperature": self.temperature_input.value(),
            # Boolean Flags
            "cache_regular": self.cache_regular_check.isChecked(),
            "run_tsp": self.run_tsp_check.isChecked(),
        }
        # Multi-value arguments
        if len(look_ahead_configs) > 0:
            params["lookahead_configs"] = " ".join(look_ahead_configs)
        if self.pregular_level_input.text().strip():
            params["pregular_level"] = self.pregular_level_input.text().strip()
        if self.plastminute_cf_input.text().strip():
            params["plastminute_cf"] = self.plastminute_cf_input.text().strip()
        if self.gurobi_param_input.text().strip():
            params["gurobi_param"] = self.gurobi_param_input.text().strip()
        if self.hexaly_param_input.text().strip():
            params["hexaly_param"] = self.hexaly_param_input.text().strip()
        return params
