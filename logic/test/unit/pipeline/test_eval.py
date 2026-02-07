"""Tests for evaluation pipeline."""

import torch
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset

from logic.src.pipeline.features.evaluate import evaluate_policy
from logic.src.pipeline.features.evaluators import GreedyEval, SamplingEval


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

        # Mock dataset
        self.dataset = TensorDataset(torch.randn(2, 5))
        self.loader = DataLoader(self.dataset, batch_size=2)

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

    def test_dispatcher(self):
        """Verify evaluate_policy dispatcher."""
        metrics = evaluate_policy(self.policy, self.env, self.loader, method="greedy", progress=False)
        assert "avg_reward" in metrics
