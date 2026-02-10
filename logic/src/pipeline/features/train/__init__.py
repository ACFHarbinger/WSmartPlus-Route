"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import __init__
"""

from hydra.core.config_store import ConfigStore

from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.models.attention_model.policy import AttentionModelPolicy
from logic.src.pipeline.rl.common.trainer import WSTrainer

from .engine import run_training
from .hpo import run_hpo
from .model_factory import create_model

# Register configuration for Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

__all__ = ["run_training", "run_hpo", "create_model", "get_env", "WSTrainer", "AttentionModelPolicy"]
