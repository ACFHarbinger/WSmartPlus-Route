"""
Tab initialization and management for MainWindow.
"""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...tabs import (
    EvalDataBatchingTab,
    EvalDecodingTab,
    EvalIOTab,
    EvalProblemTab,
    FileSystemCryptographyTab,
    FileSystemDeleteTab,
    FileSystemUpdateTab,
    GenDataAdvancedTab,
    GenDataGeneralTab,
    GenDataProblemTab,
    HyperParamOptimParserTab,
    InputAnalysisTab,
    MetaRLTrainParserTab,
    OutputAnalysisTab,
    RLCostsTab,
    RLDataTab,
    RLModelTab,
    RLOptimizerTab,
    RLOutputTab,
    RLTrainingTab,
    RunScriptsTab,
    TestSimAdvancedTab,
    TestSimIOTab,
    TestSimPolicyParamsTab,
    TestSimSettingsTab,
    TestSuiteTab,
)


class TabManager:
    """
    Manages the creation and mapping of tabs.
    """

    def __init__(self):
        self.train_tabs_map = {
            "Data": RLDataTab(),
            "Model": RLModelTab(),
            "Training": RLTrainingTab(),
            "Optimizer": RLOptimizerTab(),
            "Cost Weights": RLCostsTab(),
            "Output": RLOutputTab(),
            "Hyper-Parameter Optimization": HyperParamOptimParserTab(),
            "Meta-Learning": MetaRLTrainParserTab(),
        }
        self.gen_data_tabs_map = {
            "General Output": GenDataGeneralTab(),
            "Problem Definition": GenDataProblemTab(),
            "Advanced Settings": GenDataAdvancedTab(),
        }

        settings_tab = TestSimSettingsTab()
        io_tab = TestSimIOTab(settings_tab=settings_tab)
        self.test_sim_tabs_map = {
            "Simulator Settings": settings_tab,
            "Policy Parameters": TestSimPolicyParamsTab(),
            "IO Settings": io_tab,
            "Advanced Settings": TestSimAdvancedTab(),
        }

        self.eval_tabs_map = {
            "IO Settings": EvalIOTab(),
            "Data Configurations": EvalDataBatchingTab(),
            "Decoding Strategy": EvalDecodingTab(),
            "Problem Definition": EvalProblemTab(),
        }

        self.analysis_tabs_map = {
            "Input Analysis": InputAnalysisTab(),
            "Output Analysis": OutputAnalysisTab(),
        }

        self.file_system_tabs_map = {
            "Update Settings": FileSystemUpdateTab(),
            "Delete Settings": FileSystemDeleteTab(),
            "Cryptography Settings": FileSystemCryptographyTab(),
        }

        self.other_tabs_map = {
            "Execute Script": RunScriptsTab(),
            "Program Test Suite": TestSuiteTab(),
        }

        self.all_tabs = {
            "Train Model": self.train_tabs_map,
            "Generate Data": self.gen_data_tabs_map,
            "Evaluate": self.eval_tabs_map,
            "Test Simulator": self.test_sim_tabs_map,
            "Data Analysis": self.analysis_tabs_map,
            "File System Tools": self.file_system_tabs_map,
            "Other Tools": self.other_tabs_map,
        }

    def register_tabs(self, mediator):
        """Register all tabs with the mediator."""
        for command, tabs in self.all_tabs.items():
            for name, tab in tabs.items():
                mediator.register_tab(command, name, tab)

    def setup_tabs_in_widget(self, tab_widget, command):
        """Setup the tab widget for a specific command."""
        while tab_widget.count() > 0:
            tab_widget.removeTab(0)

        if command in self.all_tabs:
            tab_set = self.all_tabs[command]
            for title, tab_instance in tab_set.items():
                tab_widget.addTab(tab_instance, title)
        else:
            placeholder = QWidget()
            layout = QVBoxLayout(placeholder)
            layout.addWidget(QLabel(f"GUI for '{command}' coming soon."))
            tab_widget.addTab(placeholder, "Info")
