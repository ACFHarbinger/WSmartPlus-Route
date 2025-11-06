import multiprocessing as mp

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox, QLabel, QWidget,
    QSpinBox, QHBoxLayout, QPushButton,
    QScrollArea, QVBoxLayout, QSizePolicy,
    QLineEdit, QFormLayout, QDoubleSpinBox,
)
from backend.src.gui.app_definitions import HOP_METHODS, HOP_METRICS
from .components import ClickableHeaderWidget


class HyperParamOptimParserTab(QWidget):
    """
    Tab for configuring Hyperparameter Optimization (HPO) arguments.
    """
    def __init__(self):
        super().__init__()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        form_layout = QFormLayout(content)
        form_layout.addRow(QLabel("<b>General Settings</b>"))

        # --hop_method
        self.hop_method_combo = QComboBox()
        self.hop_method_combo.addItems(HOP_METHODS.keys())
        self.hop_method_combo.setCurrentText('')
        form_layout.addRow(QLabel("Optimization Method:"), self.hop_method_combo)
        
        # --hop_range (nargs='+')
        self.hop_range_input = QLineEdit("0.0 2.0")
        self.hop_range_input.setPlaceholderText("Min-Max values (space separated)")
        form_layout.addRow(QLabel("Hyper-Parameter Range:"), self.hop_range_input)
        
        # --hop_epochs
        self.hop_epochs_input = QSpinBox(minimum=1, maximum=50, value=7)
        form_layout.addRow(QLabel("Optimization Epochs:"), self.hop_epochs_input)

        # --metric
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(HOP_METRICS.keys())
        self.metric_combo.setCurrentText('Validation Loss')
        form_layout.addRow(QLabel("Metric to Optimize:"), self.metric_combo)

        # --- Ray Tune Framework Settings ---
        form_layout.addRow(QLabel("<b>Ray Tune Framework Settings</b>"))
        
        # --cpu_cores
        system_cpu_cores = mp.cpu_count()
        self.cpu_cores_input = QSpinBox(minimum=1, maximum=system_cpu_cores, value=1)
        form_layout.addRow(QLabel("CPU Cores:"), self.cpu_cores_input)

        # --verbose
        self.verbose_input = QSpinBox(minimum=0, maximum=3, value=2)
        form_layout.addRow(QLabel("Verbose Level (0-3):"), self.verbose_input)

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
        # --train_best (action='store_true', default=True) -> Checkbox should control False
        self.train_best_check = QPushButton("Train final model with best hyper-parameters")
        self.train_best_check.setChecked(True)
        self.train_best_check.setCheckable(True)
        self.train_best_check.setStyleSheet(start_green_style)
        form_layout.addRow(QLabel("Train Best Model:"), self.train_best_check)

        start_red_style = """
            QPushButton:checked {
                background-color: #06402B;
                color: white;
            }
            QPushButton {
                background-color: #8B0000;
                color: white;
            }
        """
        # --local_mode
        self.local_mode_check = QPushButton("Run Ray in Local Mode")
        self.local_mode_check.setCheckable(True)
        self.local_mode_check.setChecked(False)
        self.local_mode_check.setStyleSheet(start_red_style)
        form_layout.addRow(QLabel("Local Mode:"), self.local_mode_check)
        
        # --num_samples
        self.num_samples_input = QSpinBox(minimum=1, maximum=1000, value=20)
        form_layout.addRow(QLabel("Number of Samples:"), self.num_samples_input)

        # --- Bayesian Optimization (BO/Optuna) ---
        form_layout.addRow(QLabel("<b>Bayesian Optimization (BO)</b>"))

        # --n_trials
        self.n_trials_input = QSpinBox(minimum=1, maximum=500, value=20)
        form_layout.addRow(QLabel("Number of Trials:"), self.n_trials_input)

        # --- Timeout (Custom Header) ---
        # 1. Create a container widget for the header using the custom clickable class
        self.timeout_header_widget = ClickableHeaderWidget(self._toggle_timeout)
        self.timeout_header_widget.setStyleSheet(
            "QWidget { border: none; padding: 0; margin-top: 5px; }"
        )

        to_header_layout = QHBoxLayout(self.timeout_header_widget)
        to_header_layout.setContentsMargins(0, 0, 0, 0)
        to_header_layout.setSpacing(5)

        # 2. The main text (Standard QLabel)
        self.timeout_label = QLabel("Timeout")

        # CRITICAL: Remove expanding policy so the label shrinks to fit content
        self.timeout_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Preferred
        )
        
        # Apply the initial (collapsed) styling to the QLabel
        self.timeout_label.setStyleSheet(
            "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
        )

        # 3. The clickable toggle button (only the +/- sign)
        self.timeout_toggle_button = QPushButton("+")
        self.timeout_toggle_button.setFlat(True)
        self.timeout_toggle_button.setFixedSize(QSize(20, 20))
        self.timeout_toggle_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0; border: none; background: transparent; }"
        )
        self.timeout_toggle_button.clicked.connect(self._toggle_timeout)

        # 4. Add components to the header layout
        to_header_layout.addWidget(self.timeout_label)
        to_header_layout.addStretch()
        to_header_layout.addWidget(self.timeout_toggle_button)
        
        # 5. Add the header widget to the main layout, making it span the row
        form_layout.addRow(self.timeout_header_widget)

        # 6. Create a container for the collapsible content
        self.timeout_container = QWidget()
        timeout_layout = QFormLayout(self.timeout_container)
        timeout_layout.setContentsMargins(0, 0, 0, 0)

        # 7. Add widgets to the container's layout
        # --timeout (integer)
        self.timeout_input = QLineEdit()
        self.timeout_input.setPlaceholderText("Timeout in seconds")
        timeout_layout.addRow(QLabel("Timeout (s):"), self.timeout_input)
        
        # 8. Add the content container to the main layout
        form_layout.addWidget(self.timeout_container)

        # 9. Initialize state: hidden
        self.is_timeout_visible = False
        self.timeout_container.hide()
        
        # --n_startup_trials
        self.n_startup_trials_input = QSpinBox(minimum=0, maximum=500, value=5)
        form_layout.addRow(QLabel("Startup Trials (before pruning):"), self.n_startup_trials_input)
        
        # --n_warmup_steps
        self.n_warmup_steps_input = QSpinBox(minimum=0, maximum=50, value=3)
        form_layout.addRow(QLabel("Warmup Steps (before pruning):"), self.n_warmup_steps_input)
        
        # --interval_steps
        self.interval_steps_input = QSpinBox(minimum=1, maximum=10, value=1)
        form_layout.addRow(QLabel("Pruning Interval Steps:"), self.interval_steps_input)


        # --- Distributed Evolutionary Algorithm (DEA) ---
        form_layout.addRow(QLabel("<b>Distributed Evolutionary Algorithm (DEA)</b>"))
        
        # --eta
        self.eta_input = QDoubleSpinBox(minimum=0.01, maximum=100.0, value=10.0)
        self.eta_input.setDecimals(2)
        self.eta_input.setSingleStep(0.5)
        form_layout.addRow(QLabel("Mutation Spread (eta):"), self.eta_input)
        
        # --indpb
        self.indpb_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.2)
        self.indpb_input.setDecimals(3)
        self.indpb_input.setSingleStep(0.01)
        form_layout.addRow(QLabel("Gene Mutation Probability (indpb):"), self.indpb_input)

        # --tournsize
        self.tournsize_input = QSpinBox(minimum=2, maximum=10, value=3)
        form_layout.addRow(QLabel("Tournament Size:"), self.tournsize_input)

        # --cxpb
        self.cxpb_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.7)
        self.cxpb_input.setDecimals(3)
        self.cxpb_input.setSingleStep(0.01)
        form_layout.addRow(QLabel("Crossover Probability (cxpb):"), self.cxpb_input)

        # --mutpb
        self.mutpb_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.2)
        self.mutpb_input.setDecimals(3)
        self.mutpb_input.setSingleStep(0.01)
        form_layout.addRow(QLabel("Mutation Probability (mutpb):"), self.mutpb_input)

        # --n_pop
        self.n_pop_input = QSpinBox(minimum=1, maximum=100, value=20)
        form_layout.addRow(QLabel("Population Size (n_pop):"), self.n_pop_input)

        # --n_gen
        self.n_gen_input = QSpinBox(minimum=1, maximum=100, value=10)
        form_layout.addRow(QLabel("Generations (n_gen):"), self.n_gen_input)

        # --- Hyperband Optimization (HBO) ---
        form_layout.addRow(QLabel("<b>Hyperband Optimization (HBO)</b>"))

        # --max_tres
        self.max_tres_input = QSpinBox(minimum=1, maximum=100, value=14)
        form_layout.addRow(QLabel("Maximum Trial Resources (timesteps):"), self.max_tres_input)

        # --reduction_factor
        self.reduction_factor_input = QSpinBox(minimum=2, maximum=5, value=3)
        form_layout.addRow(QLabel("Reduction Factor:"), self.reduction_factor_input)

        # --- Grid Search (GS) ---
        form_layout.addRow(QLabel("<b>Grid Search (GS)</b>"))

        # --grid (GS, nargs='+')
        self.grid_input = QLineEdit("0.0 0.5 1.0 1.5 2.0")
        self.grid_input.setPlaceholderText("Grid values (space separated floats)")
        form_layout.addRow(QLabel("Grid Search Values:"), self.grid_input)

        # --max_conc
        self.max_conc_input = QSpinBox(minimum=1, maximum=mp.cpu_count(), value=4)
        form_layout.addRow(QLabel("Maximum Concurrent Trials:"), self.max_conc_input)

        # --- Differential Evolutionary Hyperband Optimization (DEHBO) ---
        form_layout.addRow(QLabel("<b>Differential Evolutionary Hyperband Optimization (DEHBO)</b>"))
        
        # --fevals
        self.fevals_input = QSpinBox(minimum=1, maximum=1000, value=100)
        form_layout.addRow(QLabel("Function Evaluations:"), self.fevals_input)

        # --- Random Search (RS) ---
        form_layout.addRow(QLabel("<b>Random Search (RS)</b>"))
        
        # --max_failures
        self.max_failures_input = QSpinBox(minimum=1, maximum=10, value=3)
        form_layout.addRow(QLabel("Maximum Trial Failures:"), self.max_failures_input)
        
        QVBoxLayout(self).addWidget(scroll_area)
        scroll_area.setWidget(content)

    def _toggle_timeout(self):
        """Toggles the visibility of the Timeout input field and updates the +/- sign."""
        if self.is_timeout_visible: 
            self.timeout_container.hide()
            self.timeout_toggle_button.setText("+")

            # Apply dark grey border to the QLabel when collapsed
            self.timeout_label.setStyleSheet(
                "QLabel { border: 1px solid #555555; border-radius: 4px; padding: 5px; background-color: transparent; }"
            )
        else:
            self.timeout_container.show()
            self.timeout_toggle_button.setText("-")
            
            # Remove the border from the QLabel when expanded.
            self.timeout_label.setStyleSheet(
                "QLabel { border: none; padding: 5px; background-color: transparent; }"
            )
        self.is_timeout_visible = not self.is_timeout_visible

    def get_params(self):
        params = {
            # General HPO
            "hop_method": self.hop_method_combo.currentText(),
            "hop_epochs": self.hop_epochs_input.value(),
            "metric": HOP_METRICS[self.metric_combo.currentText()],
            
            # Ray Tune
            "cpu_cores": self.cpu_cores_input.value(),
            "verbose": self.verbose_input.value(),
            "train_best": self.train_best_check.isChecked(),
            "local_mode": self.local_mode_check.isChecked(),
            "num_samples": self.num_samples_input.value(),
            
            # BO/Optuna
            "n_trials": self.n_trials_input.value(),
            "n_startup_trials": self.n_startup_trials_input.value(),
            "n_warmup_steps": self.n_warmup_steps_input.value(),
            "interval_steps": self.interval_steps_input.value(),
            
            # DEA
            "eta": self.eta_input.value(),
            "indpb": self.indpb_input.value(),
            "tournsize": self.tournsize_input.value(),
            "cxpb": self.cxpb_input.value(),
            "mutpb": self.mutpb_input.value(),
            "n_pop": self.n_pop_input.value(),
            "n_gen": self.n_gen_input.value(),
            
            # HBO
            "max_tres": self.max_tres_input.value(),
            "reduction_factor": self.reduction_factor_input.value(),
            
            # Other
            "fevals": self.fevals_input.value(),
            "max_failures": self.max_failures_input.value(),
            "max_conc": self.max_conc_input.value(),
        }

        # Handle nargs='+' and optional arguments
        
        # --hop_range (nargs='+')
        hop_range_text = self.hop_range_input.text().strip()
        if hop_range_text:
            try:
                params["hop_range"] = [float(x) for x in hop_range_text.split()]
            except ValueError:
                print("Warning: hop_range must contain space-separated floats. Defaulting to None.")
                params["hop_range"] = None
        else:
            params["hop_range"] = None
        
        # --grid (nargs='+')
        grid_text = self.grid_input.text().strip()
        if grid_text:
            try:
                params["grid"] = [float(x) for x in grid_text.split()]
            except ValueError:
                print("Warning: grid must contain space-separated floats. Defaulting to None.")
                params["grid"] = None
        else:
            params["grid"] = None

        # --timeout (optional int)
        timeout_text = self.timeout_input.text().strip()
        if timeout_text:
            try:
                params["timeout"] = int(timeout_text)
            except ValueError:
                print("Warning: timeout must be an integer. Defaulting to None.")
                params["timeout"] = None
        else:
            params["timeout"] = None

        return params
