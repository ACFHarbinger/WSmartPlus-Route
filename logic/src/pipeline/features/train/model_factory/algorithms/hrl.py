from typing import Any, Dict

import pytorch_lightning as pl
from logic.src.configs import Config
from logic.src.pipeline.rl import HRLModule


def _create_hrl(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    # Improved import handling to avoid circular deps if they exist
    from logic.src.models.hrl_manager import GATLSTManager

    manager = GATLSTManager(device=cfg.device, hidden_dim=cfg.meta_rl.meta_hidden_dim)
    return HRLModule(manager=manager, worker=policy, env=env, lr=cfg.meta_rl.meta_lr)
