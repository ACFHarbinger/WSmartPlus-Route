from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.src.constants.models import DECODE_STRATEGIES


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

        # --strategy
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(list(DECODE_STRATEGIES.keys()))
        self.strategy_combo.setCurrentText("Greedy")
        form_layout.addRow("Decoding Strategy:", self.strategy_combo)

        # --beam_width (nargs='+') - Beam width or sample size
        self.beam_width_input = QLineEdit()
        self.beam_width_input.setPlaceholderText("e.g., 50 100 200 (space separated beams/samples)")
        form_layout.addRow("Beam Width / Samples:", self.beam_width_input)

        # --softmax_temperature
        self.softmax_temperature_input = QDoubleSpinBox(minimum=0.01, maximum=10.0, value=1.0)
        self.softmax_temperature_input.setDecimals(2)
        self.softmax_temperature_input.setSingleStep(0.1)
        form_layout.addRow("Softmax Temperature:", self.softmax_temperature_input)

        scroll_area.setWidget(content)
        QVBoxLayout(self).addWidget(scroll_area)

    def get_params(self):
        # beam_width is nargs='+', so it needs special handling for output
        beam_width_str = self.beam_width_input.text().strip()

        params = {
            "decoding.strategy": DECODE_STRATEGIES.get(self.strategy_combo.currentText(), ""),
            "decoding.temperature": self.softmax_temperature_input.value(),
            "decoding.beam_width": beam_width_str or None,
        }
        return {k: v for k, v in params.items() if v is not None}
