
import torch
import pytest
from logic.src.utils.functions import boolmask

class TestBoolMask:
    def test_pad_mask(self):
        # Case 1: No padding needed (size divisible by 8)
        mask = torch.ones(8, dtype=torch.uint8)
        padded, d = boolmask._pad_mask(mask)
        assert padded.size(-1) == 8
        assert d == 1
        assert (padded == mask).all()

        # Case 2: Padding needed (size 5)
        mask = torch.ones(5, dtype=torch.uint8)
        padded, d = boolmask._pad_mask(mask)
        assert padded.size(-1) == 8  # Next multiple of 8
        assert d == 1
        assert (padded[:5] == 1).all()
        assert (padded[5:] == 0).all()

    def test_bool2long_roundtrip(self):
        # Create random boolean mask
        torch.manual_seed(42)
        size = 20
        mask = torch.rand(size) > 0.5

        # Convert to long
        long_mask = boolmask.mask_bool2long(mask.to(torch.uint8))

        # Convert back to bool
        restored_mask = boolmask.mask_long2bool(long_mask, n=size)

        assert (mask == restored_mask).all()

    def test_long_scatter(self):
        size = 128
        # Use batch size 1
        n_nodes = 100
        bool_mask = torch.zeros(n_nodes, dtype=torch.uint8).unsqueeze(0) # (1, 100)
        long_mask = boolmask.mask_bool2long(bool_mask) # (1, 2)

        # Set index 5 and 70
        idx1 = torch.tensor([5]) # (1,)
        idx2 = torch.tensor([70]) # (1,)

        # mask_long_scatter(mask, values)
        # mask shape: (1, 2), values shape: (1,) -> batch dims match ()? No.
        # mask.size()[:-1] is (1,). values.size() is (1,). Match.

        new_mask = boolmask.mask_long_scatter(long_mask.clone(), idx1)

        # Verify
        restored = boolmask.mask_long2bool(new_mask, n=n_nodes)
        assert restored[0, 5] == 1
        assert restored.sum() == 1

        # Update again
        new_mask = boolmask.mask_long_scatter(new_mask, idx2)
        restored = boolmask.mask_long2bool(new_mask, n=n_nodes)
        assert restored[0, 5] == 1
        assert restored[0, 70] == 1
        assert restored.sum() == 2

    def test_check_unset_assertion(self):
        n_nodes = 10
        bool_mask = torch.zeros(n_nodes, dtype=torch.uint8).unsqueeze(0)
        long_mask = boolmask.mask_bool2long(bool_mask)
        idx = torch.tensor([5])

        new_mask = boolmask.mask_long_scatter(long_mask, idx)

        # Try setting it again with check_unset=True
        with pytest.raises(AssertionError):
            boolmask.mask_long_scatter(new_mask, idx, check_unset=True)

        # Should pass with check_unset=False
        new_mask_2 = boolmask.mask_long_scatter(new_mask, idx, check_unset=False)
        assert (new_mask == new_mask_2).all()
