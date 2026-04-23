"""DR-ALNS (Deep Reinforcement Learning - Adaptive Large Neighborhood Search).

This package implements the hybrid approach combining PPO-based deep RL
with ALNS for online control of operator selection and parameter configuration
(AAAI 2024).

Attributes:
    DRALNSSolver: Primary solver orchestrating the search loop.
    DRALNSPPOAgent: Neural controller for meta-heuristic selection.
    DRALNSState: Feature transformation logic for search context.
    PPOTrainer: Training coordinator for the PPO agent.

Example:
    >>> from logic.src.models.core.dr_alns import DRALNSSolver
"""

from .dr_alns_solver import DRALNSSolver as DRALNSSolver
from .ppo_agent import DRALNSPPOAgent as DRALNSPPOAgent
from .ppo_agent import DRALNSState as DRALNSState
from .ppo_trainer import PPOTrainer as PPOTrainer

__all__ = ["DRALNSSolver", "DRALNSPPOAgent", "DRALNSState", "PPOTrainer"]
