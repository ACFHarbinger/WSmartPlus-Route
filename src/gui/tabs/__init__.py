from .analysis import DataAnalysisTab, OutputAnalysisTab
from .evaluation import EvalProblemTab, EvalDataBatchingTab, EvalDecodingTab, EvalIOTab
from .file_system import FileSystemCryptographyTab, FileSystemDeleteTab, FileSystemUpdateTab
from .test_simulator import TestSimAdvancedTab, TestSimSettingsTab, TestSimIOTab, TestSimPolicyParamsTab
from .generate_data import GenDataGeneralTab, GenDataProblemTab, GenDataAdvancedTab
from .reinforcement_learning import (
    RLCostsTab, RLDataTab, RLModelTab,
    RLOptimizerTab, RLOutputTab, RLTrainingTab,
)
from .hyperparam_optim import HyperParamOptimParserTab
from .meta_rl_train import MetaRLTrainParserTab
from .scripts import RunScriptsTab
from .test_suite import TestSuiteTab