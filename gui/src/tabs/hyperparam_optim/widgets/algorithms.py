"""
Settings widget for different HPO algorithms (BO, DEA, HBO, GS, DEHBO, RS).
"""

import multiprocessing as mp

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from .timeout import TimeoutWidget


class AlgorithmSettingsWidget(QWidget):
    """
    Widget containing settings for various hyperparameter optimization algorithms.
    """

    def __init__(self):
        """
        Initialize AlgorithmSettingsWidget and create form layout for all algorithms.
        """
        super().__init__()
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Bayesian Optimization (BO/Optuna) ---
        layout.addRow(QLabel("<b>Bayesian Optimization (BO)</b>"))

        self.n_trials_input = QSpinBox(minimum=1, maximum=500, value=20)
        layout.addRow(QLabel("Number of Trials:"), self.n_trials_input)

        # Timeout Widget (Inserted here as per original UI)
        self.timeout_widget = TimeoutWidget()
        layout.addRow(self.timeout_widget)

        self.n_startup_trials_input = QSpinBox(minimum=0, maximum=500, value=5)
        layout.addRow(QLabel("Startup Trials (before pruning):"), self.n_startup_trials_input)

        self.n_warmup_steps_input = QSpinBox(minimum=0, maximum=50, value=3)
        layout.addRow(QLabel("Warmup Steps (before pruning):"), self.n_warmup_steps_input)

        self.interval_steps_input = QSpinBox(minimum=1, maximum=10, value=1)
        layout.addRow(QLabel("Pruning Interval Steps:"), self.interval_steps_input)

        # --- Distributed Evolutionary Algorithm (DEA) ---
        layout.addRow(QLabel("<b>Distributed Evolutionary Algorithm (DEA)</b>"))

        self.eta_input = QDoubleSpinBox(minimum=0.01, maximum=100.0, value=10.0)
        self.eta_input.setDecimals(2)
        self.eta_input.setSingleStep(0.5)
        layout.addRow(QLabel("Mutation Spread (eta):"), self.eta_input)

        self.indpb_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.2)
        self.indpb_input.setDecimals(3)
        self.indpb_input.setSingleStep(0.01)
        layout.addRow(QLabel("Gene Mutation Probability (indpb):"), self.indpb_input)

        self.tournsize_input = QSpinBox(minimum=2, maximum=10, value=3)
        layout.addRow(QLabel("Tournament Size:"), self.tournsize_input)

        self.cxpb_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.7)
        self.cxpb_input.setDecimals(3)
        self.cxpb_input.setSingleStep(0.01)
        layout.addRow(QLabel("Crossover Probability (cxpb):"), self.cxpb_input)

        self.mutpb_input = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.2)
        self.mutpb_input.setDecimals(3)
        self.mutpb_input.setSingleStep(0.01)
        layout.addRow(QLabel("Mutation Probability (mutpb):"), self.mutpb_input)

        self.n_pop_input = QSpinBox(minimum=1, maximum=100, value=20)
        layout.addRow(QLabel("Population Size (n_pop):"), self.n_pop_input)

        self.n_gen_input = QSpinBox(minimum=1, maximum=100, value=10)
        layout.addRow(QLabel("Generations (n_gen):"), self.n_gen_input)

        # --- Hyperband Optimization (HBO) ---
        layout.addRow(QLabel("<b>Hyperband Optimization (HBO)</b>"))

        self.max_tres_input = QSpinBox(minimum=1, maximum=100, value=14)
        layout.addRow(QLabel("Maximum Trial Resources (timesteps):"), self.max_tres_input)

        self.reduction_factor_input = QSpinBox(minimum=2, maximum=5, value=3)
        layout.addRow(QLabel("Reduction Factor:"), self.reduction_factor_input)

        # --- Grid Search (GS) ---
        layout.addRow(QLabel("<b>Grid Search (GS)</b>"))

        self.grid_input = QLineEdit("0.0 0.5 1.0 1.5 2.0")
        self.grid_input.setPlaceholderText("Grid values (space separated floats)")
        layout.addRow(QLabel("Grid Search Values:"), self.grid_input)

        self.max_conc_input = QSpinBox(minimum=1, maximum=mp.cpu_count(), value=4)
        layout.addRow(QLabel("Maximum Concurrent Trials:"), self.max_conc_input)

        # --- Differential Evolutionary Hyperband Optimization (DEHBO) ---
        layout.addRow(QLabel("<b>Differential Evolutionary Hyperband Optimization (DEHBO)</b>"))

        self.fevals_input = QSpinBox(minimum=1, maximum=1000, value=100)
        layout.addRow(QLabel("Function Evaluations:"), self.fevals_input)

        # --- Random Search (RS) ---
        layout.addRow(QLabel("<b>Random Search (RS)</b>"))

        self.max_failures_input = QSpinBox(minimum=1, maximum=10, value=3)
        layout.addRow(QLabel("Maximum Trial Failures:"), self.max_failures_input)

    def get_params(self):
        """
        Extract and return all algorithm-specific parameters from the UI.

        Returns:
            dict: Dictionary of parameters for all supported optimization methods.
        """
        params = {
            # BO/Optuna
            "n_trials": self.n_trials_input.value(),
            "n_startup_trials": self.n_startup_trials_input.value(),
            "n_warmup_steps": self.n_warmup_steps_input.value(),
            "interval_steps": self.interval_steps_input.value(),
            "timeout": self.timeout_widget.get_value(),
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

        return params
