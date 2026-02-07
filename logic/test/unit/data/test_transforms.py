"""Tests for data augmentation transforms."""

import torch
import pytest
from tensordict import TensorDict
from logic.src.data.transforms import (
    batchify,
    dihedral_8_augmentation,
    symmetric_augmentation,
    StateAugmentation,
)


def test_batchify():
    """Verify TensorDict batching."""
    td = TensorDict({"a": torch.randn(4, 10, 2)}, batch_size=[4])
    num_samples = 8
    td_batch = batchify(td, num_samples)
    assert td_batch.batch_size == torch.Size([32])
    assert td_batch["a"].shape == (32, 10, 2)


def test_dihedral_8_augmentation_shape():
    """Verify Dihedral-8 output shape."""
    xy = torch.randn(4, 10, 2)
    aug_xy = dihedral_8_augmentation(xy)
    assert aug_xy.shape == (32, 10, 2)


def test_symmetric_augmentation_shape():
    """Verify symmetric augmentation output shape."""
    xy = torch.randn(32, 10, 2)
    # symmetric_augmentation expect (batch, ...) and uses internal num_augment for identity logic
    aug_xy = symmetric_augmentation(xy, num_augment=8)
    assert aug_xy.shape == (32, 10, 2)


def test_state_augmentation_call():
    """Verify StateAugmentation on TensorDict."""
    batch_size = 4
    num_augment = 8
    td = TensorDict({
        "locs": torch.rand(batch_size, 10, 2),
        "depot": torch.rand(batch_size, 1, 2)
    }, batch_size=[batch_size])

    # Test with dihedral8
    augmenter = StateAugmentation(num_augment=num_augment, augment_fn="dihedral8", feats=["locs", "depot"])
    td_aug = augmenter(td)

    assert td_aug.batch_size == torch.Size([batch_size * num_augment])
    assert td_aug["locs"].shape == (batch_size * num_augment, 10, 2)
    assert td_aug["depot"].shape == (batch_size * num_augment, 1, 2)

    # Verify identity if first_aug_identity=True
    # idx 0, 8, 16, 24 should be original
    assert torch.allclose(td_aug["locs"][0], td["locs"][0])
    assert torch.allclose(td_aug["locs"][8], td["locs"][1])


def test_state_augmentation_symmetric():
    """Verify StateAugmentation with symmetric function."""
    batch_size = 2
    num_augment = 4
    td = TensorDict({
        "locs": torch.rand(batch_size, 5, 2),
    }, batch_size=[batch_size])

    augmenter = StateAugmentation(num_augment=num_augment, augment_fn="symmetric", first_aug_identity=True)
    td_aug = augmenter(td)

    assert td_aug.batch_size == torch.Size([batch_size * num_augment])
    # check identity
    assert torch.allclose(td_aug["locs"][0], td["locs"][0])
    # others should be different (likely)
    assert not torch.allclose(td_aug["locs"][1], td["locs"][0])
