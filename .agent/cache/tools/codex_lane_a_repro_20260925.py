"""Focused review reproductions; run from the 70e660b03 worktree root."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from logic.src.data.datasets import TensorDictDataset
from logic.src.pipeline.features.eval.engine import _eval_dataset
from logic.src.pipeline.features.train.engine import _build_stage_config
from logic.src.pipeline.rl.common.base.data import _get_eval_graphs
from logic.src.pipeline.rl.common.baselines import RolloutBaseline
from logic.src.pipeline.rl.common.epoch import _build_visited_mask, prepare_epoch
from logic.src.pipeline.rl.core.reinforce import REINFORCE
from logic.src.utils.model.loader import _parse_hydra_config, load_model


def main():
    module = REINFORCE(
        env=SimpleNamespace(name="vrpp"), policy=torch.nn.Linear(1, 1),
        baseline="exponential", exp_beta=0.123, bl_alpha=0.012,
        bl_warmup_epochs=3,
    )
    assert module.baseline.beta == 0.8
    assert module.hparams["kwargs"]["exp_beta"] == 0.123
    print("B-codex-01: requested beta=.123/warmup=3; actual beta=.8/no warmup")

    critic_module = REINFORCE(
        env=SimpleNamespace(name="vrpp"), policy=torch.nn.Linear(1, 1), baseline="critic"
    )
    assert critic_module.baseline.critic is None
    assert not critic_module.baseline.get_learnable_parameters()
    print("B-codex-02: configured critic is None, zero trainable baseline parameters")

    module.log = lambda *args, **kwargs: None
    module._current_baseline_val = torch.tensor([[1.0], [3.0]])
    rewards = torch.tensor([2.0, 5.0])
    likelihood = torch.tensor([0.2, 0.7], requires_grad=True)
    loss = module.calculate_loss(
        TensorDict({}, [2]), {"reward": rewards, "log_likelihood": likelihood}, 0
    )
    assert torch.isclose(loss, torch.tensor(-1.05))
    print("B-codex-03: wrapped baseline loss=-1.05; per-instance expected=-.8")

    policy = torch.nn.Linear(1, 1)
    baseline = RolloutBaseline()
    baseline.setup(policy)
    seen = []
    baseline._rollout = lambda candidate, ds, env: seen.append(candidate is policy) or torch.zeros(len(ds))
    dataset = TensorDictDataset(TensorDict({"x": torch.zeros(2, 1)}, [2]))
    prepare_epoch(policy, SimpleNamespace(), baseline, dataset, 0)
    assert seen == [True]
    print("B-codex-04: epoch baseline wrapper uses live policy, not frozen copy")

    mask = _build_visited_mask([torch.tensor([[2], [1]])], 2, 2, torch.device("cpu"))
    assert mask.tolist() == [[False, False, True], [False, True, False]]
    print("B-codex-05: traversal [sample1->node2,sample0->node1] updates dataset rows [0->2,1->1]")

    cfg = OmegaConf.create({"task": "train", "train": {"env": {
        "curriculum_graphs": [{"num_loc": 20, "n_samples": 64}],
        "eval_graphs": [{"num_loc": 50, "n_samples": 32}],
    }}})
    stage = _build_stage_config(cfg, cfg.train.env.curriculum_graphs[0], 0)
    assert stage.train.env.graph.n_samples == 64
    assert _get_eval_graphs(stage) == []
    calls = []
    def generate(batch_size):
        calls.append(batch_size)
        return TensorDict({"x": torch.zeros(batch_size, 1)}, [batch_size])
    module.cfg = stage
    module.env = SimpleNamespace(generator=generate)
    module.setup("fit")
    assert calls == [10, 512]
    print("B-codex-06: requested train=64/eval=32; generated train=10/eval=512; eval graphs ignored")

    class Problem:
        NAME = "vrpp"
        @staticmethod
        def get_costs(*args):
            return torch.tensor([-7.0]), {
                "length": torch.tensor([1.0]), "waste": torch.tensor([1.0]),
                "overflows": torch.tensor([0.0]),
            }, None
    class Model:
        problem = Problem()
        def to(self, device): return self
        def eval(self): return self
        def set_strategy(self, *args, **kwargs): pass
    eval_cfg = SimpleNamespace(eval=SimpleNamespace(
        decoding=SimpleNamespace(strategy="greedy", temperature=1, beam_width=1),
        eval_batch_size=1, compress_mask=False, max_calc_batch_size=1,
    ), tracking=SimpleNamespace(no_progress_bar=True))
    with patch("logic.src.pipeline.features.eval.engine.evaluate_policy", return_value={
        "rewards": torch.tensor([7.0]), "sequences": torch.tensor([[0, 1, 0]]), "duration": 1.0,
    }):
        results = _eval_dataset(Model(), [{"locs": torch.zeros(2, 2)}], 1, 1, eval_cfg, torch.device("cpu"))
    assert results[0]["cost"] == 7.0
    assert np.trim_zeros(np.array([0, 1, 0, 2, 0])).tolist() == [1, 0, 2]
    print("B-codex-07: eval writes reward +7 as cost, while problem cost is -7; internal depots preserved")

    args = _parse_hydra_config({"env": {"name": "vrpp"}, "model": {
        "encoder": {"embed_dim": 32, "hidden_dim": 64, "n_layers": 1, "n_heads": 2,
                    "normalization": {"norm_type": "batch", "epsilon": 0.123}},
        "decoder": {"type": "glimpse", "n_layers": 1},
    }})
    with patch("logic.src.utils.model.loader.os.path.isfile", return_value=True), \
         patch("logic.src.utils.model.loader._load_hyperparameters", return_value=args), \
         patch("logic.src.utils.model.loader.torch_load_cpu", return_value={}), \
         patch("logic.src.models.AttentionModel") as constructor:
        load_model("/tmp/codex-probe-empty.pt")
        supplied = constructor.call_args.kwargs
        assert "norm_config" not in supplied and supplied["norm_eps_alpha"] == .123
        assert "normalization" in supplied
        constructor.return_value.load_state_dict.assert_called_once_with({}, strict=False)
    print("B-codex-08: loader passes legacy normalization kwargs; no norm_config; empty weights accepted")


if __name__ == "__main__":
    main()
