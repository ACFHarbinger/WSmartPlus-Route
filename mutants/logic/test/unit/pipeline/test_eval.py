"""Tests for evaluation pipeline."""

import torch
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset

from logic.src.pipeline.features.eval.evaluate import evaluate_policy
from logic.src.pipeline.features.eval.evaluators import GreedyEval, SamplingEval, AugmentationEval


class TestEvaluation:
    """Tests for evaluation dispatch and classes."""

    def setup_method(self):
        self.env = MagicMock()
        # Mock env reset returns batch
        self.env.reset.side_effect = lambda x: x

        self.policy = MagicMock()
        # Mock policy forward returns dict with reward
        self.policy.return_value = {
            "reward": torch.tensor([-1.0, -2.0]), # Batch 2
        }

        # Mock dataset with TensorDict
        from tensordict import TensorDict
        # We simulate what a real data loader would return: a TensorDict with batch dimension
        self.td_batch = TensorDict({
            "locs": torch.randn(2, 5, 2),
        }, batch_size=[2])

        self.loader = MagicMock()
        # Mocking __iter__ to return a list containing one TensorDict batch
        self.loader.__iter__.return_value = iter([self.td_batch])

    def test_greedy_eval(self):
        """Verify GreedyEval runs."""
        evaluator = GreedyEval(self.env, progress=False)
        metrics = evaluator(self.policy, self.loader)

        assert "avg_reward" in metrics
        assert metrics["avg_reward"] == -1.5
        self.policy.set_decode_type.assert_called_with("greedy")

    def test_sampling_eval(self):
        """Verify SamplingEval runs."""
        evaluator = SamplingEval(self.env, samples=10, progress=False)

        # Sampling returns [batch, samples] usually?
        self.policy.return_value = {
            "reward": torch.tensor([[-1.0, -1.2], [-2.0, -2.5]]), # [2, 2]
        }

        metrics = evaluator(self.policy, self.loader)
        assert "avg_reward" in metrics
        # Max over samples: -1.0 and -2.0. Mean -1.5
        assert metrics["avg_reward"] == -1.5
        self.policy.set_decode_type.assert_called_with("sampling")

    def test_augmentation_eval(self):
        """Verify AugmentationEval runs."""
        # Policy is called with augmented batch (batch_rep=num_augment)
        evaluator = AugmentationEval(self.env, num_augment=4, progress=False)

        # Batch 2, Augment 4 => Total 8 samples
        self.policy.return_value = {
            "reward": torch.tensor([-1.0, -1.2, -2.0, -2.5, -0.5, -0.8, -3.0, -3.5]),
        }

        metrics = evaluator(self.policy, self.loader)
        assert "avg_reward" in metrics
        # Max per sample:
        # Sample 1 (idx 0-3): -1.0
        # Sample 2 (idx 4-7): -0.5
        # Avg: -0.75
        assert metrics["avg_reward"] == -0.75

    def test_dispatcher(self):
        """Verify evaluate_policy dispatcher."""
        metrics = evaluate_policy(self.policy, self.env, self.loader, method="greedy", progress=False)
        assert "avg_reward" in metrics
