from PySide6.QtWidgets import (
    QComboBox, QLabel, QWidget,
    QSpinBox, QScrollArea, QVBoxLayout,
    QLineEdit, QFormLayout, QDoubleSpinBox,
)
from src.gui.app_definitions import (
    CB_EXPLORATION_METHODS,
    RWA_MODELS, RWA_OPTIMIZERS,
    MRL_METHODS, AGGREGATION_FUNCTIONS,
)


class MetaRLTrainParserTab(QWidget):
    """
    Tab for configuring Meta-Reinforcement Learning (MRL) arguments.
    """
    def __init__(self):
        super().__init__()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        form_layout = QFormLayout(content)
        form_layout.addRow(QLabel("<b>General Settings</b>"))

        # --mrl_method
        self.mrl_method_combo = QComboBox()
        self.mrl_method_combo.addItems(MRL_METHODS.keys())
        self.mrl_method_combo.setCurrentText('')
        form_layout.addRow(QLabel("Meta-Learning Method:"), self.mrl_method_combo)
        
        # --mrl_history
        self.mrl_history_input = QSpinBox(minimum=1, maximum=100, value=10)
        form_layout.addRow(QLabel("History Length (Days/Epochs):"), self.mrl_history_input)
        
        # --mrl_range (nargs='+')
        self.mrl_range_input = QLineEdit("0.01 5.0")
        self.mrl_range_input.setPlaceholderText("Min-Max values (space separated)")
        form_layout.addRow(QLabel("Dynamic Hyper-Parameter Range:"), self.mrl_range_input)
        
        # --mrl_exploration_factor
        self.mrl_exploration_factor_input = QDoubleSpinBox(minimum=0.01, maximum=10.0, value=2.0)
        self.mrl_exploration_factor_input.setDecimals(3)
        self.mrl_exploration_factor_input.setSingleStep(0.1)
        form_layout.addRow(QLabel("Exploration Factor:"), self.mrl_exploration_factor_input)

        # --mrl_lr
        self.mrl_lr_input = QDoubleSpinBox(minimum=1e-6, maximum=1.0, value=0.001)
        self.mrl_lr_input.setDecimals(6)
        self.mrl_lr_input.setSingleStep(0.0001)
        form_layout.addRow(QLabel("Learning Rate:"), self.mrl_lr_input)

        # --- Temporal Difference Learning (TDL) ---
        form_layout.addRow(QLabel("<b>Temporal Difference Learning (TDL)</b>"))
        
        # --tdl_lr_decay
        self.tdl_lr_decay_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=1.0)
        self.tdl_lr_decay_input.setDecimals(5)
        self.tdl_lr_decay_input.setSingleStep(0.001)
        form_layout.addRow(QLabel("Learning Rate Decay:"), self.tdl_lr_decay_input)

        # --- Contextual Bandits (CB) ---
        form_layout.addRow(QLabel("<b>Contextual Bandits (CB)</b>"))

        # --cb_exploration_method
        self.cb_exploration_method_combo = QComboBox()
        self.cb_exploration_method_combo.addItems(CB_EXPLORATION_METHODS.keys())
        self.cb_exploration_method_combo.setCurrentText('Upper Confidence Bound (UCB)')
        form_layout.addRow(QLabel("Exploration Method:"), self.cb_exploration_method_combo)
        
        # --cb_num_configs
        self.cb_num_configs_input = QSpinBox(minimum=1, maximum=100, value=10)
        form_layout.addRow(QLabel("Weight Configs:"), self.cb_num_configs_input)

        # --cb_context_features (nargs='+')
        self.cb_context_features_input = QLineEdit("waste overflow length visited_ratio day")
        self.cb_context_features_input.setPlaceholderText("space separated list of features")
        form_layout.addRow(QLabel("Context Features:"), self.cb_context_features_input)

        # --cb_features_aggregation
        self.cb_features_aggregation_combo = QComboBox()
        self.cb_features_aggregation_combo.addItems(AGGREGATION_FUNCTIONS.keys())
        self.cb_features_aggregation_combo.setCurrentText('Average')
        form_layout.addRow(QLabel("Feature Aggregation:"), self.cb_features_aggregation_combo)

        # --cb_epsilon_decay
        self.cb_epsilon_decay_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.995)
        self.cb_epsilon_decay_input.setDecimals(5)
        self.cb_epsilon_decay_input.setSingleStep(0.001)
        form_layout.addRow(QLabel("Epsilon Decay:"), self.cb_epsilon_decay_input)
        
        # --cb_min_epsilon
        self.cb_min_epsilon_input = QDoubleSpinBox(minimum=0.0, maximum=0.5, value=0.01)
        self.cb_min_epsilon_input.setDecimals(5)
        self.cb_min_epsilon_input.setSingleStep(0.001)
        form_layout.addRow(QLabel("Mininimum Epsilon:"), self.cb_min_epsilon_input)

        # --- Multi-Objective Reinforcement Learning (MORL) ---
        form_layout.addRow(QLabel("<b>Multi-Objective Reinforcement Learning (MORL)</b>"))

        # --morl_objectives (nargs='+')
        self.morl_objectives_input = QLineEdit("waste_efficiency overflow_rate")
        self.morl_objectives_input.setPlaceholderText("space separated list of objectives")
        form_layout.addRow(QLabel("Objectives:"), self.morl_objectives_input)

        # --morl_adaptation_rate
        self.morl_adaptation_rate_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.1)
        self.morl_adaptation_rate_input.setDecimals(3)
        self.morl_adaptation_rate_input.setSingleStep(0.01)
        form_layout.addRow(QLabel("Adaptation Rate:"), self.morl_adaptation_rate_input)

        # --- Reward Weight Adjustment (RWA) ---
        form_layout.addRow(QLabel("<b>Reward Weight Adjustment (RWA)</b>"))

        # --rwa_model
        self.rwa_model_combo = QComboBox()
        self.rwa_model_combo.addItems(RWA_MODELS.keys())
        self.rwa_model_combo.setCurrentText('Recurrent Neural Network (RNN)')
        form_layout.addRow(QLabel("Model:"), self.rwa_model_combo)

        # --rwa_optimizer
        self.rwa_optimizer_combo = QComboBox()
        self.rwa_optimizer_combo.addItems(RWA_OPTIMIZERS.keys())
        self.rwa_optimizer_combo.setCurrentText('Root Mean Square Propagation (RMSProp)')
        form_layout.addRow(QLabel("Optimizer:"), self.rwa_optimizer_combo)

        # --rwa_embedding_dim
        self.rwa_embedding_dim_input = QSpinBox(minimum=16, maximum=512, value=128)
        self.rwa_embedding_dim_input.setSingleStep(16)
        form_layout.addRow(QLabel("Embedding Dim:"), self.rwa_embedding_dim_input)

        # --rwa_batch_size
        self.rwa_batch_size_input = QSpinBox(minimum=1, maximum=1024, value=256)
        self.rwa_batch_size_input.setSingleStep(32)
        form_layout.addRow(QLabel("Batch Size:"), self.rwa_batch_size_input)

        # --rwa_step
        self.rwa_step_input = QSpinBox(minimum=1, maximum=1000, value=100)
        form_layout.addRow(QLabel("Model Update Step:"), self.rwa_step_input)
        
        # --rwa_update_step
        self.rwa_update_step_input = QSpinBox(minimum=1, maximum=1000, value=100)
        form_layout.addRow(QLabel("Weight Update Step:"), self.rwa_update_step_input)

        QVBoxLayout(self).addWidget(scroll_area)
        scroll_area.setWidget(content)

    def get_params(self):
        # 1. Start with the values that do not require dictionary lookup
        params = {
            "mrl_history": self.mrl_history_input.value(),
            "mrl_exploration_factor": self.mrl_exploration_factor_input.value(),
            "mrl_lr": self.mrl_lr_input.value(),
            "tdl_lr_decay": self.tdl_lr_decay_input.value(),
            "cb_num_configs": self.cb_num_configs_input.value(),
            "cb_epsilon_decay": self.cb_epsilon_decay_input.value(),
            "cb_min_epsilon": self.cb_min_epsilon_input.value(),
            "morl_adaptation_rate": self.morl_adaptation_rate_input.value(),
            "rwa_embedding_dim": self.rwa_embedding_dim_input.value(),
            "rwa_batch_size": self.rwa_batch_size_input.value(),
            "rwa_step": self.rwa_step_input.value(),
            "rwa_update_step": self.rwa_update_step_input.value(),
        }

        # 2. Handle QComboBox Lookups 
        # Helper function for safe lookup
        def get_cli_value(combo_box, lookup_dict):
            text = combo_box.currentText()
            # Look up the short CLI value (e.g., 'ucb') using the full UI text
            cli_arg = lookup_dict.get(text, None)
            
            # If a match is found, return it. Otherwise, return the raw text if it exists.
            return cli_arg if cli_arg else text if text else None

        # --mrl_method
        mrl_method_arg = get_cli_value(self.mrl_method_combo, MRL_METHODS)
        if mrl_method_arg:
            params["mrl_method"] = mrl_method_arg

        # --cb_exploration_method
        cb_exploration_arg = get_cli_value(self.cb_exploration_method_combo, CB_EXPLORATION_METHODS)
        if cb_exploration_arg:
            params["cb_exploration_method"] = cb_exploration_arg

        # --cb_features_aggregation
        cb_agg_arg = get_cli_value(self.cb_features_aggregation_combo, AGGREGATION_FUNCTIONS)
        if cb_agg_arg:
            params["cb_features_aggregation"] = cb_agg_arg

        # --rwa_model
        rwa_model_arg = get_cli_value(self.rwa_model_combo, RWA_MODELS)
        if rwa_model_arg:
            params["rwa_model"] = rwa_model_arg

        # --rwa_optimizer
        rwa_optim_arg = get_cli_value(self.rwa_optimizer_combo, RWA_OPTIMIZERS)
        if rwa_optim_arg:
            params["rwa_optimizer"] = rwa_optim_arg
            
        # 3. Handle nargs='+' arguments (your existing logic, moved for clarity)
        # --mrl_range (List of Floats)
        mrl_range_text = self.mrl_range_input.text().strip()
        if mrl_range_text:
            try:
                # The assumption here is that your CLI handles this list of floats.
                # If your main window preview logic handles lists correctly, this is fine.
                params["mrl_range"] = [float(x) for x in mrl_range_text.split()]
            except ValueError:
                print("Warning: mrl_range must contain space-separated floats. Skipping.")
        
        # --cb_context_features (List of Strings)
        cb_context_features_text = self.cb_context_features_input.text().strip()
        if cb_context_features_text:
            params["cb_context_features"] = cb_context_features_text.split()

        # --morl_objectives (List of Strings)
        morl_objectives_text = self.morl_objectives_input.text().strip()
        if morl_objectives_text:
            params["morl_objectives"] = morl_objectives_text.split()

        return params
