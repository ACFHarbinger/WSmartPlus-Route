"""Tests for attention heatmap capture and WandB logging (§A.2 Option C)."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from logic.src.pipeline.callbacks.pytorch.attention_heatmaps import AttentionHeatmapCallback
from logic.src.tracking.logging.visualization.heatmaps import (
    capture_runtime_attention,
    extract_attention_matrix,
    maybe_log_eval_attention_heatmaps,
    render_attention_heatmap_png,
)
from torch import nn


def test_extract_attention_matrix_4d():
    tensor = torch.ones(2, 4, 5, 5)
    tensor[0, 1] = torch.eye(5)
    mat = extract_attention_matrix(tensor, head_idx=1, batch_idx=0)
    assert mat is not None
    assert mat.shape == (5, 5)
    assert np.allclose(mat.sum(axis=-1), np.ones(5), atol=1e-5)


def test_extract_attention_matrix_3d_heads():
    tensor = torch.zeros(8, 6, 6)
    tensor[2] = torch.eye(6)
    mat = extract_attention_matrix(tensor, head_idx=2)
    assert mat is not None
    assert mat.shape == (6, 6)


def test_render_attention_heatmap_png(tmp_path):
    matrix = np.random.rand(4, 4)
    path = str(tmp_path / "attn.png")
    render_attention_heatmap_png(matrix, "Test", path)
    assert (tmp_path / "attn.png").is_file()


def test_capture_runtime_attention_with_hooks():
    class FakeAttn(nn.Module):
        def forward(self, x, mask=None):
            self.last_attn = (torch.softmax(torch.eye(4).unsqueeze(0), dim=-1), mask)
            return x

    class FakeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.att = nn.Module()
            self.att.module = FakeAttn()

        def forward(self, x, mask=None):
            return self.att.module(x, mask=mask)

    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer()])

        def forward(self, x, mask=None):
            return self.layers[0](x)

    class FakePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedder = FakeEncoder()
            self.problem = MagicMock()

        def forward(self, batch, env=None, strategy="greedy"):
            h = batch["loc"]
            self.embedder(h)
            return {"reward": torch.tensor(0.0)}

    policy = FakePolicy()
    batch = {"loc": torch.randn(1, 4, 2)}
    mats = capture_runtime_attention(policy, batch)
    assert len(mats) == 1
    assert mats[0][0] == 0
    assert mats[0][1].shape == (4, 4)


@patch("logic.src.tracking.logging.visualization.heatmaps.wandb")
def test_maybe_log_eval_attention_heatmaps(mock_wandb, tmp_path):
    mock_wandb.run = MagicMock()
    mock_wandb.Image = MagicMock(side_effect=lambda p, caption=None: p)

    tracking = MagicMock()
    tracking.log_attention = True
    tracking.log_attention_heatmaps = False
    tracking.log_dir = str(tmp_path)
    tracking.wandb_mode = "online"

    cfg = MagicMock()
    cfg.tracking = tracking

    policy = MagicMock()
    policy.parameters.return_value = iter([torch.zeros(1)])
    policy.eval.return_value = policy

    with patch(
        "logic.src.tracking.logging.visualization.heatmaps.capture_runtime_attention",
        return_value=[(0, np.eye(3))],
    ):
        paths = maybe_log_eval_attention_heatmaps(policy, {"loc": torch.zeros(1, 3, 2)}, cfg)

    assert len(paths) == 1
    assert paths[0].endswith("runtime_layer0.png")
    assert mock_wandb.log.called


def test_attention_heatmap_callback_skips_when_disabled():
    cb = AttentionHeatmapCallback(tracking_cfg=None)
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = 0
    with patch(
        "logic.src.pipeline.callbacks.pytorch.attention_heatmaps.maybe_log_eval_attention_heatmaps"
    ) as mock_log:
        cb.on_validation_epoch_end(trainer, MagicMock())
        mock_log.assert_not_called()


def test_attention_heatmap_callback_runs_when_enabled():
    tracking = MagicMock()
    tracking.log_attention = True
    tracking.log_attention_heatmaps = False
    tracking.viz_every_n_epochs = 0

    cb = AttentionHeatmapCallback(tracking_cfg=tracking)
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = 1
    trainer.global_step = 10
    trainer.loggers = []

    class _FakeLoader:
        def __iter__(self):
            yield {"loc": torch.zeros(1, 3, 2)}

    trainer.val_dataloaders = _FakeLoader()

    with patch(
        "logic.src.pipeline.callbacks.pytorch.attention_heatmaps.maybe_log_eval_attention_heatmaps"
    ) as mock_log:
        cb.on_validation_epoch_end(trainer, MagicMock())
        mock_log.assert_called_once()
