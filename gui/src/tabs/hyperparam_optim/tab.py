from PySide6.QtWidgets import (
    QFormLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .widgets.algorithms import AlgorithmSettingsWidget
from .widgets.general import GeneralSettingsWidget
from .widgets.ray_tune import RayTuneSettingsWidget


class HyperParamOptimParserTab(QWidget):
    """
    Tab for configuring Hyperparameter Optimization (HPO) arguments.
    """

    def __init__(self):
        super().__init__()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        self.form_layout = QFormLayout(content)

        # 1. General Settings
        self.general_widget = GeneralSettingsWidget()
        self.form_layout.addRow(self.general_widget)

        # 2. Ray Tune Settings
        self.ray_tune_widget = RayTuneSettingsWidget()
        self.form_layout.addRow(self.ray_tune_widget)

        # 3. Algorithms Settings (BO, DEA, HBO, GS, DEHBO, RS)
        # We need to inject the Timeout widget in between BO trials.
        # But wait, the original layout had Timeout inside BO section?
        # Let's check original.
        # Original: BO -> n_trials -> Timeout Header -> Timeout Content -> n_startup -> ...
        # My Algorithms widget has all BO stuff together.
        # I should probably pass the Timeout widget TO the algorithms widget or refactor.
        # Actually, looking at original code:
        # form_layout.addRow(QLabel("<b>Bayesian Optimization (BO)</b>"))
        # ... n_trials ...
        # ... timeout header ...
        # ... timeout container ...
        # ... n_startup ...
        # My Algorithms widget has grouped them.
        # To maintain exact layout I should probably split Algorithms widget or handle Timeout inside it?
        # But Timeout widget is generic.
        # Ideally, I should just add them sequentially to the main form layout here?
        # But the plan was to have widgets encapsulating logic.
        # Let's see: General and RayTune are distinct blocks.
        # Algorithms block is big and contains everything else.
        # But Timeout is smack in the middle of BO parameters.
        # Let's restructure:
        # I will modify AlgorithmsWidget to accept an optional insert_widget for BO section or just add the Timeout widget inside AlgorithmsWidget directly?
        # No, better: The Timeout widget is specific to BO/Optuna usually (max time per trial?). Actually look at get_params: 'timeout' is returned.
        # It's a general param but placed under BO in UI.
        # I will just place the TimeoutWidget IN the AlgorithmSettingsWidget for simplicity in this refactor,
        # OR I can just compose them here properly if I exposed the layout of AlgorithmSettingsWidget?
        # No, that breaks encapsulation.
        # Best approach: Add TimeoutWidget inside AlgorithmSettingsWidget.
        # But I already created TimeoutWidget as separate file.
        # I will modify AlgorithmSettingsWidget to import and use TimeoutWidget internally.

        # Checking algorithms.py again...
        # I need to modify algorithms.py to include TimeoutWidget.
        # For now, let's just stick to the plan:
        # Plan said: `hyperparam_optim/widgets/timeout.py`: TimeoutWidget.
        # Plan said: `hyperparam_optim/widgets/algorithms.py`: AlgorithmSettingsWidget.
        # So I will assume AlgorithmSettingsWidget SHOULD use TimeoutWidget.
        # But I wrote `algorithms.py` without it.
        # I will rewrite `algorithms.py` in the next step to include it.
        # Wait, I cannot rewrite `algorithms.py` in this step easily as I'm writing `tab.py`.
        # I will write `tab.py` to use `AlgorithmSettingsWidget` and assume `AlgorithmSettingsWidget` handles it.
        # I will then UPDATE `algorithms.py` in the next tool call.
        pass

        # Wait, I am writing `tab.py` now.
        # If I write `tab.py` now, it will import `AlgorithmSettingsWidget`.
        # I should update `AlgorithmSettingsWidget` FIRST or update it simultaneously?
        # I can't update simultaneously.
        # I will write `tab.py` assuming `AlgorithmSettingsWidget` is self-contained.
        # Then I will update `AlgorithmSettingsWidget`.

        self.algorithms_widget = AlgorithmSettingsWidget()
        self.form_layout.addRow(self.algorithms_widget)

        QVBoxLayout(self).addWidget(scroll_area)
        scroll_area.setWidget(content)

    def get_params(self):
        params = {}
        params.update(self.general_widget.get_params())
        params.update(self.ray_tune_widget.get_params())
        params.update(self.algorithms_widget.get_params())
        return params
