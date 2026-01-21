"""
Hierarchical Reinforcement Learning (HRL) module.

Implements a Manager-Worker architecture:
- Manager: GATLSTManager (decides if/what to collect)
- Worker: ConstructivePolicy (decides the route)
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.gat_lstm_manager import GATLSTManager
from logic.src.models.policies.base import ConstructivePolicy


class HRLModule(pl.LightningModule):
    """
    Lightning module for Hierarchical RL.

    Coordinates a high-level manager and a low-level worker.
    """

    def __init__(
        self,
        manager: GATLSTManager,
        worker: ConstructivePolicy,
        env: RL4COEnvBase,
        lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["manager", "worker", "env"])
        self.manager = manager
        self.worker = worker
        self.env = env
        self.lr = lr

    def training_step(self, batch: TensorDict, batch_idx: int):
        """
        Combined training step for Manager and Worker.
        """
        td = self.env.reset(batch)

        # 1. Manager Decision
        # Prepare inputs for manager
        # static: locs (B, N, 2)
        # dynamic: demand history (B, N, 10) - handle if missing
        static = td["locs"]

        if "demand_history" in td.keys():
            dynamic = td["demand_history"]
        else:
            # Fallback to current demand padded if history not available
            dynamic = td["demand"].unsqueeze(-1).expand(-1, -1, 10)

        # global_features: summary stats (B, 2)
        global_features = torch.stack(
            [
                td["demand"].mean(dim=1),
                td["demand"].max(dim=1)[0],
            ],
            dim=-1,
        )

        mask_action, gate_action, manager_value = self.manager.select_action(static, dynamic, global_features)

        # 2. Worker Decision (Routing)
        # If gate_action == 1 (Dispatch), run worker
        dispatch_indices = (gate_action == 1).nonzero().squeeze(-1)

        total_reward = torch.zeros(td.batch_size, device=td.device)

        if len(dispatch_indices) > 0:
            td_worker = td[dispatch_indices]

            # Apply manager's mask_action to td_worker['visited']
            td_worker["visited"] = td_worker["visited"] | (mask_action[dispatch_indices] == 0)

            out_worker = self.worker(td_worker, self.env)
            total_reward[dispatch_indices] = out_worker["reward"]

        # Log metrics
        self.log("train/manager_gate_rate", gate_action.float().mean())
        self.log("train/manager_visit_rate", mask_action.float().mean())
        self.log("train/reward", total_reward.mean(), prog_bar=True)

        return None

    def configure_optimizers(self):
        params = list(self.manager.parameters()) + list(self.worker.parameters())
        return torch.optim.Adam(params, lr=self.lr)
