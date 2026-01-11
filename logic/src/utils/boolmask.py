import torch
import torch.nn.functional as F


"""
Boolean Mask Utilities.

This module provides optimized functions for converting between different
mask representations (boolean, byte, long) and performing masked scatter operations
efficiently on GPU.
"""


# Attention, Learn to Solve Routing Problems
def _pad_mask(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


def _mask_bool2byte(mask):
    """
    Converts boolean mask to byte packed format (8 bools -> 1 byte).

    Args:
        mask (Tensor): Boolean mask (uint8).

    Returns:
        Tensor: Packed byte mask.
    """
    assert mask.dtype == torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(
        -1, dtype=torch.uint8
    )


def _mask_byte2long(mask):
    """
    Converts byte mask to long representation by treating as bits.

    Args:
        mask (Tensor): Byte mask.

    Returns:
        Tensor: Long tensor.
    """
    assert mask.dtype == torch.uint8
    mask, d = _pad_mask(mask)
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (
        mask.view(*mask.size()[:-1], d, 8).long()
        << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def mask_bool2long(mask):
    """
    Converts boolean mask directly to long integers representation.

    Args:
        mask (Tensor): Boolean mask.

    Returns:
        Tensor: Long mask.
    """
    assert mask.dtype == torch.uint8
    return _mask_byte2long(_mask_bool2byte(mask))


def _mask_long2byte(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is None:
        n = 8 * mask.size(-1)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def _mask_byte2bool(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is None:
        n = 8 * mask.size(-1)
    return (
        mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)
    ).view(*mask.size()[:-1], -1)[..., :n] > 0


def mask_long2bool(mask, n=None):
    """
    Converts long representation back to boolean mask.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions.
    If values contains -1, nothing is set.
    Note: does not work for setting multiple values at once (like normal scatter).

    Args:
        mask (Tensor): Mask to update (long).
        values (Tensor): Indices to set to 1.
        check_unset (bool): Check if bit is already set.

    Returns:
        Tensor: Updated mask.
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))
