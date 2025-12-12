from PySide6.QtWidgets import (
    QFormLayout, QVBoxLayout,
    QScrollArea, QLabel, QWidget,
    QComboBox, QLineEdit, QDoubleSpinBox,
)
from gui.src.utils.app_definitions import DECODE_STRATEGIES, DECODE_TYPES


class EvalDecodingTab(QWidget):
    """
    Tab for controlling the inference/search strategy (greedy, sampling, beam search).
    """
    def __init__(self):
        super().__init__()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        form_layout = QFormLayout(content)

        form_layout.addRow(QLabel("<b>Decoding Parameters</b>"))

        # --decode_strategy
        self.decode_strategy_combo = QComboBox()
        self.decode_strategy_combo.addItems(DECODE_STRATEGIES.keys())
        self.decode_strategy_combo.setCurrentText('Greedy')
        form_layout.addRow("Decode Strategy:", self.decode_strategy_combo)
        
        # --decode_type (Used to specify the type of decoding for the output)
        self.decode_type_combo = QComboBox()
        self.decode_type_combo.addItems([dt.title() for dt in DECODE_TYPES])
        self.decode_type_combo.setCurrentText('Greedy')
        form_layout.addRow("Decode Type (Output):", self.decode_type_combo)

        # --width (nargs='+') - Beam width or sample size
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("e.g., 50 100 200 (space separated beams/samples)")
        form_layout.addRow("Beam Width / Samples:", self.width_input)
        
        # --softmax_temperature
        self.softmax_temperature_input = QDoubleSpinBox(minimum=0.01, maximum=10.0, value=1.0)
        self.softmax_temperature_input.setDecimals(2)
        self.softmax_temperature_input.setSingleStep(0.1)
        form_layout.addRow("Softmax Temperature:", self.softmax_temperature_input)

        scroll_area.setWidget(content)
        QVBoxLayout(self).addWidget(scroll_area)

    def get_params(self):
        # width is nargs='+', so it needs special handling for output
        width_str = self.width_input.text().strip()
        
        params = {
            "decode_strategy": DECODE_STRATEGIES.get(self.decode_strategy_combo.currentText(), ""),
            "decode_type": self.decode_type_combo.currentText().lower(),
            "softmax_temperature": self.softmax_temperature_input.value(),
            "width": width_str or None,
        }
        return {k: v for k, v in params.items() if v is not None}
