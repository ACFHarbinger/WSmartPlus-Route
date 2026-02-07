from hydra.core.config_store import ConfigStore

from logic.src.configs import Config

from .engine import run_training
from .hpo import run_hpo
from .model_factory import create_model

# Register configuration for Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

__all__ = ["run_training", "run_hpo", "create_model"]
