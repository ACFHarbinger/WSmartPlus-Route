"""Unit tests for runtime attention ring-buffer (§A.2 Option A)."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from logic.src.tracking.attention_buffer import (
    AttentionRingBuffer,
    ensure_attention_buffer,
    install_attention_ring_buffer,
)
from logic.src.tracking.logging.modules.attention_emit import (
    ATTENTION_VIZ_MARKER,
    maybe_emit_attention_viz,
    send_attention_viz_to_gui,
)
from torch import nn


def test_attention_ring_buffer_record_and_cap():
    buffer = AttentionRingBuffer(max_history=2, max_matrix_dim=8)
    tensor = torch.softmax(torch.eye(4).unsqueeze(0).unsqueeze(0), dim=-1)
    buffer.record(0, tensor, head_idx=0)
    buffer.record(1, tensor, head_idx=0)
    buffer.record(2, tensor, head_idx=0)
    assert len(buffer) == 2
    assert buffer.get_snapshots()[0]["layer"] == 1


def test_attention_ring_buffer_decode_step():
    buffer = AttentionRingBuffer()
    tensor = torch.softmax(torch.eye(3).unsqueeze(0), dim=-1)
    buffer.record(0, tensor)
    buffer.bump_decode_step()
    buffer.record(0, tensor)
    steps = [s["decode_step"] for s in buffer.get_snapshots()]
    assert steps == [0, 1]


def test_install_attention_ring_buffer_hooks():
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

    buffer = AttentionRingBuffer()
    encoder = FakeEncoder()
    handles = install_attention_ring_buffer(encoder, buffer)
    encoder.layers[0](torch.randn(1, 4, 8))
    assert len(buffer) == 1
    for handle in handles:
        handle.remove()


def test_ensure_attention_buffer_attaches_to_policy():
    class FakeAttn(nn.Module):
        def forward(self, x, mask=None):
            self.last_attn = (torch.softmax(torch.eye(3).unsqueeze(0), dim=-1), mask)
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

    class FakePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedder = FakeEncoder()

    policy = FakePolicy()
    buffer = ensure_attention_buffer(policy)
    assert buffer is not None
    policy.embedder.layers[0](torch.randn(1, 3, 8))
    assert len(buffer) == 1


def test_send_attention_viz_to_gui_writes_jsonl(tmp_path, capsys):
    snapshots = [{"layer": 0, "head": 0, "decode_step": 0, "n_nodes": 3, "matrix": [[1.0, 0, 0]] * 3}]
    log_path = str(tmp_path / "attention_viz.jsonl")
    send_attention_viz_to_gui(snapshots, "val", 1, 10, log_path)
    captured = capsys.readouterr().out
    assert ATTENTION_VIZ_MARKER in captured
    content = (tmp_path / "attention_viz.jsonl").read_text()
    assert ATTENTION_VIZ_MARKER in content
    assert '"phase":"val"' in content.replace(" ", "")


def test_maybe_emit_attention_viz_respects_tracking_flag():
    policy = MagicMock()
    policy.attention_buffer = AttentionRingBuffer()
    policy.attention_buffer.record(
        0,
        torch.softmax(torch.eye(3).unsqueeze(0), dim=-1),
    )
    model = MagicMock()
    model.policy = policy

    tracking_off = MagicMock()
    tracking_off.log_attention = False
    cfg_off = MagicMock()
    cfg_off.tracking = tracking_off
    assert maybe_emit_attention_viz(model, cfg_off) is False

    tracking_on = MagicMock()
    tracking_on.log_attention = True
    tracking_on.log_dir = "logs"
    cfg_on = MagicMock()
    cfg_on.tracking = tracking_on

    with patch(
        "logic.src.tracking.logging.modules.attention_emit.send_attention_viz_to_gui"
    ) as mock_send:
        assert maybe_emit_attention_viz(model, cfg_on, phase="eval", epoch=2, step=5) is True
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert args[1] == "eval"
        assert args[2] == 2
        assert args[3] == 5
