"""Tests for boolean mask utilities."""

import pytest
import torch

from logic.src.utils.functions.boolmask import (
    _mask_bool2byte,
    _mask_byte2bool,
    _pad_mask,
    mask_bool2long,
    mask_long2bool,
    mask_long_scatter,
)


def test_pad_mask():
    """Verify mask padding to multiples of 8."""
    # Already divisible by 8
    mask8 = torch.zeros(8, dtype=torch.uint8)
    padded8, bytes8 = _pad_mask(mask8)
    assert padded8.size(0) == 8
    assert bytes8 == 1

    # Needs padding
    mask10 = torch.zeros(10, dtype=torch.uint8)
    padded16, bytes16 = _pad_mask(mask10)
    assert padded16.size(0) == 16
    assert bytes16 == 2

    # Empty
    mask0 = torch.zeros(0, dtype=torch.uint8)
    padded0, bytes0 = _pad_mask(mask0)
    assert padded0.size(0) == 0
    assert bytes0 == 0


def test_bool_byte_conversion():
    """Verify roundtrip between boolean and byte representations."""
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=torch.uint8)
    packed = _mask_bool2byte(mask)
    # 10 bits -> 2 bytes (16 bits)
    assert packed.shape == (2,)

    unpacked = _mask_byte2bool(packed, n=10)
    assert unpacked.shape == (10,)
    assert torch.equal(unpacked, mask.bool())


def test_bool_long_conversion():
    """Verify roundtrip between boolean and long representations."""
    # Test batching
    mask = torch.randint(0, 2, (4, 15), dtype=torch.uint8)
    long_mask = mask_bool2long(mask)
    # 15 bits -> 2 bytes -> 1 long (long is 8 bytes, so 1 long is plenty for 15 bits)
    # mask_byte2long pads to move to long
    assert long_mask.dtype == torch.int64

    recovered = mask_long2bool(long_mask, n=15)
    assertRecovered = recovered.to(torch.uint8)
    assert torch.equal(mask, assertRecovered)


def test_mask_long_scatter():
    """Verify scatter operations on long masks."""
    batch = 4
    num_longs = 2  # 128 bits
    mask = torch.zeros(batch, num_longs, dtype=torch.int64)

    # Set bit 10 in batch 0, bit 70 in batch 1
    values = torch.tensor([10, 70, -1, 5], dtype=torch.int64)

    # -1 should not set anything (per docstring "If values contains -1, nothing is set")
    # Actually looking at code: values_ = values[..., None]. where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # if values is -1, where will be 0 for all rng since rng >= 0.

    updated = mask_long_scatter(mask, values, check_unset=True)

    # Batch 0: bit 10 is set in first long
    assert (updated[0, 0] & (1 << 10)) > 0
    # Batch 1: bit 70 is set in second long (70 % 64 = 6)
    assert (updated[1, 1] & (1 << 6)) > 0
    # Batch 2: nothing set
    assert updated[2].sum() == 0
    # Batch 3: bit 5 set in first long
    assert (updated[3, 0] & (1 << 5)) > 0


def test_mask_long_scatter_check_unset():
    """Verify check_unset logic."""
    mask = torch.zeros(1, 1, dtype=torch.int64)
    mask[0, 0] = 1 << 5

    # Should fail if trying to set bit 5 again with check_unset=True
    with pytest.raises(AssertionError):
        mask_long_scatter(mask, torch.tensor([5], dtype=torch.int64), check_unset=True)

    # Should work with check_unset=False
    mask_long_scatter(mask, torch.tensor([5], dtype=torch.int64), check_unset=False)
