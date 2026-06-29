"""Unit tests for cleanup scripts (remove_hpo.py, remove_meta.py, remove_eval.py, remove_callbacks.py)."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from logic.src.utils.package.remove_callbacks import (
    clean_callbacks_init as callbacks_clean_init,
)
from logic.src.utils.package.remove_callbacks import (
    clean_trainer as callbacks_clean_trainer,
)
from logic.src.utils.package.remove_eval import (
    clean_hydra_dispatch as eval_clean_dispatch,
)
from logic.src.utils.package.remove_hpo import (
    clean_configs_init as hpo_clean_configs,
)
from logic.src.utils.package.remove_hpo import (
    clean_tasks_init as hpo_clean_tasks,
)
from logic.src.utils.package.remove_meta import (
    clean_rl_init as meta_clean_rl,
)


class TestRemovals(unittest.TestCase):
    """Test cases for checking operations of removal/cleanup scripts."""

    def test_hpo_cleanups(self) -> None:
        """Test file modifications in HPO removal script."""
        # 1. Test clean_tasks_init
        mock_tasks_init = MagicMock(spec=Path)
        mock_tasks_init.exists.return_value = True
        mock_tasks_init.read_text.return_value = (
            "from .hpo import HPOConfig\n"
            "from .sim import SimConfig\n"
            "__all__ = [\n"
            '    "HPOConfig",\n'
            '    "SimConfig",\n'
            "]\n"
        )
        hpo_clean_tasks(mock_tasks_init, ["HPOConfig"])
        mock_tasks_init.write_text.assert_called_once()
        written = mock_tasks_init.write_text.call_args[0][0]
        self.assertIn("# from .hpo import HPOConfig  # AUTO-REMOVED", written)
        self.assertNotIn('"HPOConfig",', written)
        self.assertIn('"SimConfig",', written)

        # 2. Test clean_configs_init
        mock_configs_init = MagicMock(spec=Path)
        mock_configs_init.exists.return_value = True
        mock_configs_init.read_text.return_value = (
            "from .tasks import DataConfig, HPOConfig, SimConfig\n"
            "    hpo: HPOConfig = field(default_factory=HPOConfig)\n"
            "    sim: SimConfig = field(default_factory=SimConfig)\n"
            "__all__ = [\n"
            '    "HPOConfig",\n'
            '    "SimConfig",\n'
            "]\n"
        )
        hpo_clean_configs(mock_configs_init, ["HPOConfig"], ["hpo"])
        mock_configs_init.write_text.assert_called_once()
        written_config = mock_configs_init.write_text.call_args[0][0]
        self.assertIn("from .tasks import DataConfig, SimConfig", written_config)
        self.assertNotIn("hpo: HPOConfig = field(default_factory=HPOConfig)", written_config)
        self.assertIn("sim: SimConfig = field(default_factory=SimConfig)", written_config)
        self.assertNotIn('"HPOConfig",', written_config)
        self.assertIn('"SimConfig",', written_config)

    def test_meta_cleanups(self) -> None:
        """Test file modifications in Meta-RL removal script."""
        # 1. Test clean_rl_init
        mock_rl_init = MagicMock(spec=Path)
        mock_rl_init.exists.return_value = True
        mock_rl_init.read_text.return_value = (
            "from logic.src.pipeline.rl.core import (\n"
            "    A2C,\n"
            "    MetaRLModule,\n"
            "    HRLModule,\n"
            ")\n"
            "RL_ALGORITHM_REGISTRY = {\n"
            '    "a2c": A2C,\n'
            '    "meta_rl": MetaRLModule,\n'
            '    "hrl": HRLModule,\n'
            "}\n"
            "__all__ = [\n"
            '    "A2C",\n'
            '    "MetaRLModule",\n'
            '    "HRLModule",\n'
            "]\n"
        )
        meta_clean_rl(mock_rl_init)
        mock_rl_init.write_text.assert_called_once()
        written_rl = mock_rl_init.write_text.call_args[0][0]
        self.assertNotIn("MetaRLModule", written_rl)
        self.assertNotIn("HRLModule", written_rl)
        self.assertNotIn("meta_rl", written_rl)
        self.assertNotIn("hrl", written_rl)

    def test_eval_cleanups(self) -> None:
        """Test file modifications in Eval removal script."""
        # Test clean_hydra_dispatch for eval
        mock_dispatch = MagicMock(spec=Path)
        mock_dispatch.exists.return_value = True
        mock_dispatch.read_text.return_value = (
            '    if task == "eval":\n'
            "        from logic.src.pipeline.features.eval import run_evaluate_model\n"
            "        run_evaluate_model(cfg)\n"
            "        return 0.0\n"
            '    if task == "test_sim":\n'
            "        return 0.0\n"
        )
        eval_clean_dispatch(mock_dispatch)
        mock_dispatch.write_text.assert_called_once()
        written_dispatch = mock_dispatch.write_text.call_args[0][0]
        self.assertNotIn('task == "eval"', written_dispatch)
        self.assertIn('task == "test_sim"', written_dispatch)

    def test_callbacks_cleanups(self) -> None:
        """Test file modifications in Callbacks removal script."""
        # 1. Test clean_callbacks_init
        mock_cb_init = MagicMock(spec=Path)
        mock_cb_init.exists.return_value = True
        mock_cb_init.read_text.return_value = (
            "from .pytorch.model_summary import ModelSummaryCallback\n"
            "from .pytorch.speed_monitor import SpeedMonitor\n"
            "__all__ = [\n"
            '    "ModelSummaryCallback",\n'
            '    "SpeedMonitor",\n'
            "]\n"
        )
        callbacks_clean_init(mock_cb_init, ["ModelSummaryCallback"])
        mock_cb_init.write_text.assert_called_once()
        written_cb = mock_cb_init.write_text.call_args[0][0]
        self.assertIn("# from .pytorch.model_summary import ModelSummaryCallback  # AUTO-REMOVED", written_cb)
        self.assertNotIn('"ModelSummaryCallback",', written_cb)
        self.assertIn('"SpeedMonitor",', written_cb)

        # 2. Test clean_trainer
        mock_trainer = MagicMock(spec=Path)
        mock_trainer.exists.return_value = True
        mock_trainer.read_text.return_value = (
            "from logic.src.pipeline.callbacks import (\n"
            "    ModelSummaryCallback,\n"
            "    TrainingDisplayCallback,\n"
            ")\n"
            "        # Add custom model summary callback\n"
            "        if ModelSummaryCallback not in callback_types:\n"
            "            callbacks.append(ModelSummaryCallback())\n"
            "        # Find if TrainingDisplayCallback exist\n"
            "        if display_callback is None:\n"
            "            callbacks.append(TrainingDisplayCallback())\n"
        )
        callbacks_clean_trainer(mock_trainer, ["ModelSummaryCallback", "TrainingDisplayCallback"])
        mock_trainer.write_text.assert_called_once()
        written_trainer = mock_trainer.write_text.call_args[0][0]
        self.assertIn("        # # Add custom model summary callback", written_trainer)
        self.assertIn("        # if ModelSummaryCallback not in callback_types:", written_trainer)
        self.assertIn("        # callbacks.append(ModelSummaryCallback())", written_trainer)
        self.assertIn("        # # Find if TrainingDisplayCallback exist", written_trainer)

    def test_enums_cleanups(self) -> None:
        """Test file modifications in Enums removal script."""
        from logic.src.utils.package.remove_enums import process_python_file

        mock_file = MagicMock(spec=Path)
        mock_file.name = "policy_hgs.py"
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = (
            "from logic.src.enums import GlobalRegistry, PolicyTag\n"
            "@GlobalRegistry.register(\n"
            "    PolicyTag.META_HEURISTIC,\n"
            ")\n"
            "class HGSPolicy:\n"
            "    pass\n"
        )
        process_python_file(mock_file)
        mock_file.write_text.assert_called_once()
        written = mock_file.write_text.call_args[0][0]
        self.assertNotIn("from logic.src.enums", written)
        self.assertNotIn("@GlobalRegistry.register", written)
        self.assertIn("class HGSPolicy:", written)

    def test_data_cleanups(self) -> None:
        """Test clean_init_file for data removal script."""
        from logic.src.utils.package.remove_data import clean_init_file

        mock_init = MagicMock(spec=Path)
        mock_init.exists.return_value = True
        mock_init.read_text.return_value = (
            "from .pytorch.baseline_dataset import BaselineDataset\n"
            "from .pytorch.fast_td_dataset import FastTdDataset\n"
            "__all__ = [\n"
            '    "BaselineDataset",\n'
            '    "FastTdDataset",\n'
            "]\n"
        )
        clean_init_file(mock_init, {"baseline_dataset"})
        mock_init.write_text.assert_called_once()
        written = mock_init.write_text.call_args[0][0]
        self.assertIn("# from .pytorch.baseline_dataset import BaselineDataset  # AUTO-REMOVED", written)
        self.assertNotIn('"BaselineDataset",', written)
        self.assertIn('"FastTdDataset"', written)

    def test_security_cleanups(self) -> None:
        """Test file modifications in Security removal script."""
        from logic.src.utils.package.remove_security import patch_setup_env, patch_google_maps

        mock_file = MagicMock(spec=Path)
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = (
            "from logic.src.utils.security import decrypt_file_data, load_key\n"
            "                    key = load_key(symkey_name=symkey_name, env_filename=env_filename or \".env\")\n"
            "                    data = decrypt_file_data(key, gplic_path)\n"
        )
        patch_setup_env(mock_file)
        mock_file.write_text.assert_called_once()
        written = mock_file.write_text.call_args[0][0]
        self.assertNotIn("from logic.src.utils.security import decrypt_file_data, load_key\n", written)
        self.assertIn("# from logic.src.utils.security", written)
        self.assertIn("raise ValueError", written)

        mock_file2 = MagicMock(spec=Path)
        mock_file2.exists.return_value = True
        mock_file2.read_text.return_value = (
            "from logic.src.utils.security import decrypt_file_data, load_key\n"
            "            sym_key = load_key(kwargs[\"symkey_name\"], kwargs[\"env_filename\"])\n"
            "            api_key = decrypt_file_data(sym_key, kwargs[\"gapik_file\"])\n"
        )
        patch_google_maps(mock_file2)
        mock_file2.write_text.assert_called_once()
        written2 = mock_file2.write_text.call_args[0][0]
        self.assertNotIn("from logic.src.utils.security import decrypt_file_data, load_key\n", written2)
        self.assertIn("# from logic.src.utils.security", written2)
        self.assertIn("raise ValueError", written2)
