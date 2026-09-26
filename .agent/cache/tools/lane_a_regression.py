"""Regression checks for the fixed lane-A bugs (inverse of codex_lane_a_repro_20260925.py).

usage (repo root): .venv/bin/python .agent/cache/tools/lane_a_regression.py
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.getcwd())

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from logic.src.data.datasets import TensorDictDataset
from logic.src.pipeline.features.eval.engine import _eval_dataset
from logic.src.pipeline.features.train.engine import _build_stage_config
from logic.src.pipeline.rl.common.base.data import _get_eval_graphs
from logic.src.pipeline.rl.common.baselines import RolloutBaseline, WarmupBaseline
from logic.src.pipeline.rl.common.epoch import prepare_epoch
from logic.src.pipeline.rl.core.reinforce import REINFORCE
from logic.src.utils.model.loader import load_model


def main():
    # DS-06 / B-codex-01: cfg.rl kwargs reach the baseline
    module = REINFORCE(env=SimpleNamespace(name="vrpp"), policy=torch.nn.Linear(1, 1),
                       baseline="exponential", exp_beta=0.123, bl_warmup_epochs=3)
    assert isinstance(module.baseline, WarmupBaseline) and module.baseline.warmup_epochs == 3
    plain = REINFORCE(env=SimpleNamespace(name="vrpp"), policy=torch.nn.Linear(1, 1),
                      baseline="exponential", exp_beta=0.123)
    assert abs(plain.baseline.beta - 0.123) < 1e-9
    print("DS-06 ok: exp_beta=0.123 and 3 warmup epochs reach the baseline")

    # DS-05 / B-codex-03: [B,1] baseline against [B] reward gives a per-instance advantage
    plain.log = lambda *a, **k: None
    plain._current_baseline_val = torch.tensor([[1.0], [3.0]])
    loss = plain.calculate_loss(TensorDict({}, [2]), {"reward": torch.tensor([2.0, 5.0]),
                                "log_likelihood": torch.tensor([0.2, 0.7], requires_grad=True)}, 0)
    assert torch.isclose(loss, torch.tensor(-0.8)), loss
    print("DS-05 ok: wrapped baseline loss = -0.8 (per instance)")

    # DS-08 / B-codex-04: epoch wrapping uses the frozen baseline policy
    policy = torch.nn.Linear(1, 1)
    baseline = RolloutBaseline(); baseline.setup(policy)
    seen = []
    baseline._rollout = lambda cand, ds, env: seen.append(cand is baseline.baseline_policy) or torch.zeros(len(ds))
    prepare_epoch(policy, SimpleNamespace(), baseline, TensorDictDataset(TensorDict({"x": torch.zeros(2, 1)}, [2])), 0)
    assert seen == [True]
    print("DS-08 ok: epoch baseline uses the frozen copy")

    # DS-01 / B-codex-06: stage graph sizes and eval graphs are honoured
    cfg = OmegaConf.create({"task": "train", "train": {"env": {
        "curriculum_graphs": [{"num_loc": 20, "n_samples": 64}],
        "eval_graphs": [{"num_loc": 50, "n_samples": 32}]}}})
    stage = _build_stage_config(cfg, cfg.train.env.curriculum_graphs[0], 0)
    assert len(_get_eval_graphs(stage)) == 1
    calls = []
    module.cfg = OmegaConf.create({"task": "train", "train": {"env": {"graph": {"n_samples": 64}}}})
    module.env = SimpleNamespace(generator=lambda batch_size: calls.append(batch_size) or TensorDict({"x": torch.zeros(batch_size, 1)}, [batch_size]))
    module.setup("fit")
    assert calls[0] == 64, calls
    print("DS-01 ok: train dataset uses the stage n_samples (64); eval graphs are found")

    # DS-04 / B-codex-07: eval reports cost = -reward
    class Problem:
        NAME = "vrpp"
        @staticmethod
        def get_costs(*a):
            return torch.tensor([-7.0]), {"length": torch.tensor([1.0]), "waste": torch.tensor([1.0]), "overflows": torch.tensor([0.0])}, None
    class Model:
        problem = Problem()
        def to(self, d): return self
        def eval(self): return self
        def set_strategy(self, *a, **k): pass
    eval_cfg = SimpleNamespace(eval=SimpleNamespace(decoding=SimpleNamespace(strategy="greedy", temperature=1, beam_width=1),
                               eval_batch_size=1, compress_mask=False, max_calc_batch_size=1),
                               tracking=SimpleNamespace(no_progress_bar=True))
    with patch("logic.src.pipeline.features.eval.engine.evaluate_policy",
               return_value={"rewards": torch.tensor([7.0]), "sequences": torch.tensor([[0, 1, 0]]), "duration": 1.0}):
        res = _eval_dataset(Model(), [{"locs": torch.zeros(2, 2)}], 1, 1, eval_cfg, torch.device("cpu"))
    assert res[0]["cost"] == -7.0 and res[0]["reward"] == 7.0
    print("DS-04 ok: reward +7 is reported as cost -7")

    # DS-10 / B-codex-09: empty or unknown checkpoints are rejected
    from logic.src.utils.model.loader import _parse_hydra_config
    args = _parse_hydra_config({"env": {"name": "vrpp"}, "model": {
        "encoder": {"embed_dim": 32, "hidden_dim": 64, "n_layers": 1, "n_heads": 2},
        "decoder": {"type": "glimpse", "n_layers": 1}}})
    for payload in ({}, {"state_dict": {}}):
        with patch("logic.src.utils.model.loader.os.path.isfile", return_value=True), \
             patch("logic.src.utils.model.loader._load_hyperparameters", return_value=args), \
             patch("logic.src.utils.model.loader.torch_load_cpu", return_value=payload), \
             patch("logic.src.models.AttentionModel") as ctor:
            ctor.return_value.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
            try:
                load_model("/nonexistent/epoch-1.pt")
            except ValueError as e:
                assert "checkpoint" in str(e).lower() or "parameters" in str(e).lower(), e
                continue
            raise AssertionError(f"checkpoint {payload!r} was accepted")
    print("DS-10 ok: empty / unknown checkpoints are rejected")
    print("ALL LANE-A REGRESSIONS PASS")


if __name__ == "__main__":
    main()
