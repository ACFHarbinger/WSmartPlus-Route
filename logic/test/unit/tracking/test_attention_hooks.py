"""Unit tests for the attention capture hooks in tracking hooks."""

import pytest
import torch
from logic.src.tracking.hooks.attention_hooks import add_attention_hooks
from torch import nn


class DummyAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_attn = None

    def forward(self, x):
        # Emulate a tuple of (weights, mask)
        self.last_attn = (torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0]))
        return x


class DummyAttWrapper:
    def __init__(self, m):
        self.module = m


class DummyLayer:
    def __init__(self, att_module):
        self.att = DummyAttWrapper(att_module)


class DummyModel(nn.Module):
    def __init__(self, att_module):
        super().__init__()
        self.att_module = att_module
        self.layers = [DummyLayer(att_module)]

    def forward(self, x):
        return self.att_module(x)


@pytest.mark.unit
def test_attention_hooks():
    att_module = DummyAttentionModule()
    model = DummyModel(att_module)

    hook_data = add_attention_hooks(model)
    assert len(hook_data["handles"]) == 1

    # Before forward pass, weights/masks should be empty
    assert len(hook_data["weights"]) == 0
    assert len(hook_data["masks"]) == 0

    # Perform forward pass
    x = torch.tensor([1.0])
    model(x)

    # After forward pass, weights/masks should be captured
    assert len(hook_data["weights"]) == 1
    assert len(hook_data["masks"]) == 1
    assert torch.equal(hook_data["weights"][0], torch.tensor([1.0, 2.0]))
    assert torch.equal(hook_data["masks"][0], torch.tensor([0.0, 1.0]))

    # Test removing handle
    for handle in hook_data["handles"]:
        handle.remove()
