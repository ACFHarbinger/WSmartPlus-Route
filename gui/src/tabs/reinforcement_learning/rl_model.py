from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.src.utils.app_definitions import (
    ACTIVATION_FUNCTIONS,
    AGGREGATION_FUNCTIONS,
    ENCODERS,
    MODELS,
    NORMALIZATION_METHODS,
)

from ...styles.globals import START_RED_STYLE
from .rl_base import BaseReinforcementLearningTab


class RLModelTab(BaseReinforcementLearningTab):
    """Model parameters for Reinforcement Learning"""

    def __init__(self):
        super().__init__()
        self.widgets = {}
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()

        layout = QFormLayout()

        # Model
        self.widgets["model"] = QComboBox()
        self.widgets["model"].addItems(MODELS.keys())
        layout.addRow(QLabel("Model:"), self.widgets["model"])

        # Encoder
        self.widgets["encoder"] = QComboBox()
        self.widgets["encoder"].addItems(ENCODERS.keys())
        layout.addRow(QLabel("Encoder:"), self.widgets["encoder"])

        # Embedding dim
        self.widgets["embedding_dim"] = QSpinBox()
        self.widgets["embedding_dim"].setRange(1, 1024)
        self.widgets["embedding_dim"].setValue(128)
        layout.addRow(QLabel("Embedding Dimension:"), self.widgets["embedding_dim"])

        # Hidden dim
        self.widgets["hidden_dim"] = QSpinBox()
        self.widgets["hidden_dim"].setRange(1, 2048)
        self.widgets["hidden_dim"].setValue(512)
        layout.addRow(QLabel("Hidden Dimension:"), self.widgets["hidden_dim"])

        # N encode layers
        self.widgets["n_encode_layers"] = QSpinBox()
        self.widgets["n_encode_layers"].setRange(1, 20)
        self.widgets["n_encode_layers"].setValue(3)
        layout.addRow(QLabel("Encode Layers:"), self.widgets["n_encode_layers"])

        # Temporal horizon
        self.widgets["temporal_horizon"] = QSpinBox()
        self.widgets["temporal_horizon"].setRange(0, 100)
        self.widgets["temporal_horizon"].setValue(0)
        layout.addRow(QLabel("Temporal Horizon:"), self.widgets["temporal_horizon"])

        # Tanh clipping
        self.widgets["tanh_clipping"] = QDoubleSpinBox()
        self.widgets["tanh_clipping"].setRange(0, 100)
        self.widgets["tanh_clipping"].setValue(10.0)
        layout.addRow(QLabel("Tanh Clipping:"), self.widgets["tanh_clipping"])

        # Normalization
        self.widgets["normalization"] = QComboBox()
        self.widgets["normalization"].addItems([nm.replace("_", " ").title() for nm in NORMALIZATION_METHODS])
        self.widgets["normalization"].addItem("")
        layout.addRow(QLabel("Normalization:"), self.widgets["normalization"])

        # Activation
        self.widgets["activation"] = QComboBox()
        self.widgets["activation"].addItems(ACTIVATION_FUNCTIONS.keys())
        layout.addRow(QLabel("Activation:"), self.widgets["activation"])

        # Dropout
        self.widgets["dropout"] = QDoubleSpinBox()
        self.widgets["dropout"].setRange(0, 1)
        self.widgets["dropout"].setSingleStep(0.1)
        self.widgets["dropout"].setValue(0.1)
        layout.addRow(QLabel("Dropout:"), self.widgets["dropout"])

        # Aggregation graph
        self.widgets["aggregation_graph"] = QComboBox()
        self.widgets["aggregation_graph"].addItems(sorted(AGGREGATION_FUNCTIONS.keys()))
        self.widgets["aggregation_graph"].addItem("")
        layout.addRow(QLabel("Graph Aggregation:"), self.widgets["aggregation_graph"])

        # Aggregation
        self.widgets["aggregation"] = QComboBox()
        self.widgets["aggregation"].addItems(AGGREGATION_FUNCTIONS.keys())
        layout.addRow(QLabel("Node Aggregation:"), self.widgets["aggregation"])

        # N heads
        self.widgets["n_heads"] = QSpinBox()
        self.widgets["n_heads"].setRange(1, 32)
        self.widgets["n_heads"].setValue(8)
        layout.addRow(QLabel("Attention Heads:"), self.widgets["n_heads"])

        # Mask options
        self.widgets["mask_inner"] = QPushButton("Mask Inner")
        self.widgets["mask_inner"].setCheckable(True)
        self.widgets["mask_inner"].setChecked(True)
        self.widgets["mask_inner"].setStyleSheet(START_RED_STYLE)
        layout.addRow(QLabel("Masking:"), self.widgets["mask_inner"])

        self.widgets["mask_logits"] = QPushButton("Mask Logits")
        self.widgets["mask_logits"].setCheckable(True)
        self.widgets["mask_logits"].setChecked(True)
        self.widgets["mask_logits"].setStyleSheet(START_RED_STYLE)
        layout.addRow("", self.widgets["mask_logits"])

        self.widgets["mask_graph"] = QPushButton("Mask Graph")
        self.widgets["mask_graph"].setCheckable(True)
        self.widgets["mask_graph"].setChecked(False)
        self.widgets["mask_graph"].setStyleSheet(START_RED_STYLE)
        layout.addRow("", self.widgets["mask_graph"])

        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        # Connect signals
        for widget in self.widgets.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.paramsChanged.emit())
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.paramsChanged.emit())
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(lambda: self.paramsChanged.emit())
            elif isinstance(widget, QPushButton):
                widget.toggled.connect(lambda: self.paramsChanged.emit())

    def get_params(self):
        params = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if text:
                    params[key] = text
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                if not text or text == "none":
                    continue  # Skip empty or 'none' entries

                # 1. Determine the source dictionary based on the widget's key
                # This ensures we are always looking up the correct mapping
                if key == "model":
                    lookup_dict = MODELS
                elif key == "encoder":
                    lookup_dict = ENCODERS
                elif key == "activation":
                    lookup_dict = ACTIVATION_FUNCTIONS
                elif key in ["aggregation", "aggregation_graph"]:
                    lookup_dict = AGGREGATION_FUNCTIONS
                elif key == "normalization":
                    # Normalization uses replaced strings in the UI,
                    # so we need to reverse the replacement to get the key
                    # Assuming keys are lowercase with underscores (e.g., batch_norm)
                    # and UI displays Title Case with spaces (Batch Norm)
                    original_key = text.lower().replace(" ", "_")
                    if original_key and original_key in NORMALIZATION_METHODS:
                        params[key] = original_key
                    continue
                else:
                    # For any other combobox, just pass the text if no mapping is needed
                    params[key] = text
                    continue

                # 2. Perform the lookup: Use the UI text (e.g., "Attention Model") to get the value (e.g., "attn")
                cli_argument = lookup_dict.get(text, None)

                if cli_argument:
                    # 3. Assign the command-line value to the parameter key
                    params[key] = cli_argument

        # This part handles boolean flags that might not be QCheckBoxes.
        params["mask_inner"] = self.widgets["mask_inner"].isChecked()
        params["mask_logits"] = self.widgets["mask_logits"].isChecked()
        params["mask_graph"] = self.widgets["mask_graph"].isChecked()
        return params
