"""
Boolean Mask Utilities.

This module provides optimized functions for converting between different
mask representations (boolean, byte, long) and performing masked scatter operations
efficiently on GPU.
"""

from inspect import signature as _mutmut_signature
from typing import Annotated, Callable, ClassVar

import torch
import torch.nn.functional as F

MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg=None):
    """Forward call to original or mutated function, depending on the environment"""
    import os

    mutant_under_test = os.environ["MUTANT_UNDER_TEST"]
    if mutant_under_test == "fail":
        from mutmut.__main__ import MutmutProgrammaticFailException

        raise MutmutProgrammaticFailException("Failed programmatically")
    elif mutant_under_test == "stats":
        from mutmut.__main__ import record_trampoline_hit

        record_trampoline_hit(orig.__module__ + "." + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + "." + orig.__name__ + "__mutmut_"
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition(".")[-1]
    if self_arg:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_orig(mask):
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


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_1(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = None
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_2(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) / 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_3(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = +mask.size(-1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_4(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(None) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_5(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(+1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_6(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-2) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_7(mask):
    """
    Pads mask to be divisible by 8 for byte packing.

    Args:
        mask (Tensor): Input mask.

    Returns:
        tuple: (padded_mask, number_of_bytes)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) % 9
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_8(mask):
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
    if pad == 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_9(mask):
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
    if pad != 1:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_10(mask):
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
        mask = None
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_11(mask):
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
        mask = F.pad(None, [0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_12(mask):
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
        mask = F.pad(mask, None)
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_13(mask):
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
        mask = F.pad([0, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_14(mask):
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
        mask = F.pad(
            mask,
        )
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_15(mask):
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
        mask = F.pad(mask, [1, pad])
    return mask, mask.size(-1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_16(mask):
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
    return mask, mask.size(-1) / 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_17(mask):
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
    return mask, mask.size(None) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_18(mask):
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
    return mask, mask.size(+1) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_19(mask):
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
    return mask, mask.size(-2) // 8


# Attention, Learn to Solve Routing Problems
def x__pad_mask__mutmut_20(mask):
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
    return mask, mask.size(-1) // 9


x__pad_mask__mutmut_mutants: ClassVar[MutantDict] = {
    "x__pad_mask__mutmut_1": x__pad_mask__mutmut_1,
    "x__pad_mask__mutmut_2": x__pad_mask__mutmut_2,
    "x__pad_mask__mutmut_3": x__pad_mask__mutmut_3,
    "x__pad_mask__mutmut_4": x__pad_mask__mutmut_4,
    "x__pad_mask__mutmut_5": x__pad_mask__mutmut_5,
    "x__pad_mask__mutmut_6": x__pad_mask__mutmut_6,
    "x__pad_mask__mutmut_7": x__pad_mask__mutmut_7,
    "x__pad_mask__mutmut_8": x__pad_mask__mutmut_8,
    "x__pad_mask__mutmut_9": x__pad_mask__mutmut_9,
    "x__pad_mask__mutmut_10": x__pad_mask__mutmut_10,
    "x__pad_mask__mutmut_11": x__pad_mask__mutmut_11,
    "x__pad_mask__mutmut_12": x__pad_mask__mutmut_12,
    "x__pad_mask__mutmut_13": x__pad_mask__mutmut_13,
    "x__pad_mask__mutmut_14": x__pad_mask__mutmut_14,
    "x__pad_mask__mutmut_15": x__pad_mask__mutmut_15,
    "x__pad_mask__mutmut_16": x__pad_mask__mutmut_16,
    "x__pad_mask__mutmut_17": x__pad_mask__mutmut_17,
    "x__pad_mask__mutmut_18": x__pad_mask__mutmut_18,
    "x__pad_mask__mutmut_19": x__pad_mask__mutmut_19,
    "x__pad_mask__mutmut_20": x__pad_mask__mutmut_20,
}


def _pad_mask(*args, **kwargs):
    result = _mutmut_trampoline(x__pad_mask__mutmut_orig, x__pad_mask__mutmut_mutants, args, kwargs)
    return result


_pad_mask.__signature__ = _mutmut_signature(x__pad_mask__mutmut_orig)
x__pad_mask__mutmut_orig.__name__ = "x__pad_mask"


def x__mask_bool2byte__mutmut_orig(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_1(mask):
    """
    Converts boolean mask to byte packed format (8 bools -> 1 byte).

    Args:
        mask (Tensor): Boolean mask (uint8).

    Returns:
        Tensor: Packed byte mask.
    """
    assert mask.dtype != torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_2(mask):
    """
    Converts boolean mask to byte packed format (8 bools -> 1 byte).

    Args:
        mask (Tensor): Boolean mask (uint8).

    Returns:
        Tensor: Packed byte mask.
    """
    assert mask.dtype == torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = None
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_3(mask):
    """
    Converts boolean mask to byte packed format (8 bools -> 1 byte).

    Args:
        mask (Tensor): Boolean mask (uint8).

    Returns:
        Tensor: Packed byte mask.
    """
    assert mask.dtype == torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = _pad_mask(None)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_4(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(None, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_5(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=None)


def x__mask_bool2byte__mutmut_6(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(dtype=torch.uint8)


def x__mask_bool2byte__mutmut_7(mask):
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
        -1,
    )


def x__mask_bool2byte__mutmut_8(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) >> torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_9(mask):
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
    return (mask.view(*mask.size()[:-1], None, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_10(mask):
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
    return (mask.view(*mask.size()[:-1], d, None) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_11(mask):
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
    return (mask.view(d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_12(mask):
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
    return (mask.view(*mask.size()[:-1], 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_13(mask):
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
    return (
        mask.view(
            *mask.size()[:-1],
            d,
        )
        << torch.arange(8, out=mask.new())
    ).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_14(mask):
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
    return (mask.view(*mask.size()[:+1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_15(mask):
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
    return (mask.view(*mask.size()[:-2], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_16(mask):
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
    return (mask.view(*mask.size()[:-1], d, 9) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_17(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(None, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_18(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=None)).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_19(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_20(mask):
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
    return (
        mask.view(*mask.size()[:-1], d, 8)
        << torch.arange(
            8,
        )
    ).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_21(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(9, out=mask.new())).sum(-1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_22(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(+1, dtype=torch.uint8)


def x__mask_bool2byte__mutmut_23(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-2, dtype=torch.uint8)


x__mask_bool2byte__mutmut_mutants: ClassVar[MutantDict] = {
    "x__mask_bool2byte__mutmut_1": x__mask_bool2byte__mutmut_1,
    "x__mask_bool2byte__mutmut_2": x__mask_bool2byte__mutmut_2,
    "x__mask_bool2byte__mutmut_3": x__mask_bool2byte__mutmut_3,
    "x__mask_bool2byte__mutmut_4": x__mask_bool2byte__mutmut_4,
    "x__mask_bool2byte__mutmut_5": x__mask_bool2byte__mutmut_5,
    "x__mask_bool2byte__mutmut_6": x__mask_bool2byte__mutmut_6,
    "x__mask_bool2byte__mutmut_7": x__mask_bool2byte__mutmut_7,
    "x__mask_bool2byte__mutmut_8": x__mask_bool2byte__mutmut_8,
    "x__mask_bool2byte__mutmut_9": x__mask_bool2byte__mutmut_9,
    "x__mask_bool2byte__mutmut_10": x__mask_bool2byte__mutmut_10,
    "x__mask_bool2byte__mutmut_11": x__mask_bool2byte__mutmut_11,
    "x__mask_bool2byte__mutmut_12": x__mask_bool2byte__mutmut_12,
    "x__mask_bool2byte__mutmut_13": x__mask_bool2byte__mutmut_13,
    "x__mask_bool2byte__mutmut_14": x__mask_bool2byte__mutmut_14,
    "x__mask_bool2byte__mutmut_15": x__mask_bool2byte__mutmut_15,
    "x__mask_bool2byte__mutmut_16": x__mask_bool2byte__mutmut_16,
    "x__mask_bool2byte__mutmut_17": x__mask_bool2byte__mutmut_17,
    "x__mask_bool2byte__mutmut_18": x__mask_bool2byte__mutmut_18,
    "x__mask_bool2byte__mutmut_19": x__mask_bool2byte__mutmut_19,
    "x__mask_bool2byte__mutmut_20": x__mask_bool2byte__mutmut_20,
    "x__mask_bool2byte__mutmut_21": x__mask_bool2byte__mutmut_21,
    "x__mask_bool2byte__mutmut_22": x__mask_bool2byte__mutmut_22,
    "x__mask_bool2byte__mutmut_23": x__mask_bool2byte__mutmut_23,
}


def _mask_bool2byte(*args, **kwargs):
    result = _mutmut_trampoline(x__mask_bool2byte__mutmut_orig, x__mask_bool2byte__mutmut_mutants, args, kwargs)
    return result


_mask_bool2byte.__signature__ = _mutmut_signature(x__mask_bool2byte__mutmut_orig)
x__mask_bool2byte__mutmut_orig.__name__ = "x__mask_bool2byte"


def x__mask_byte2long__mutmut_orig(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_1(mask):
    """
    Converts byte mask to long representation by treating as bits.

    Args:
        mask (Tensor): Byte mask.

    Returns:
        Tensor: Long tensor.
    """
    assert mask.dtype != torch.uint8
    mask, d = _pad_mask(mask)
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_2(mask):
    """
    Converts byte mask to long representation by treating as bits.

    Args:
        mask (Tensor): Byte mask.

    Returns:
        Tensor: Long tensor.
    """
    assert mask.dtype == torch.uint8
    mask, d = None
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_3(mask):
    """
    Converts byte mask to long representation by treating as bits.

    Args:
        mask (Tensor): Byte mask.

    Returns:
        Tensor: Long tensor.
    """
    assert mask.dtype == torch.uint8
    mask, d = _pad_mask(None)
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_4(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(None)


def x__mask_byte2long__mutmut_5(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() >> (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_6(mask):
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
        mask.view(*mask.size()[:-1], None, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_7(mask):
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
        mask.view(*mask.size()[:-1], d, None).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_8(mask):
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
    return (mask.view(d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)).sum(-1)


def x__mask_byte2long__mutmut_9(mask):
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
    return (mask.view(*mask.size()[:-1], 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)).sum(
        -1
    )


def x__mask_byte2long__mutmut_10(mask):
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
        mask.view(
            *mask.size()[:-1],
            d,
        ).long()
        << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_11(mask):
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
        mask.view(*mask.size()[:+1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_12(mask):
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
        mask.view(*mask.size()[:-2], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_13(mask):
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
        mask.view(*mask.size()[:-1], d, 9).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_14(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) / 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_15(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(None, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_16(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=None, device=mask.device) * 8)).sum(-1)


def x__mask_byte2long__mutmut_17(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=None) * 8)).sum(-1)


def x__mask_byte2long__mutmut_18(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(dtype=torch.int64, device=mask.device) * 8)).sum(
        -1
    )


def x__mask_byte2long__mutmut_19(mask):
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
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, device=mask.device) * 8)).sum(-1)


def x__mask_byte2long__mutmut_20(mask):
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
        << (
            torch.arange(
                8,
                dtype=torch.int64,
            )
            * 8
        )
    ).sum(-1)


def x__mask_byte2long__mutmut_21(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(9, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def x__mask_byte2long__mutmut_22(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 9)
    ).sum(-1)


def x__mask_byte2long__mutmut_23(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(+1)


def x__mask_byte2long__mutmut_24(mask):
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
        mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-2)


x__mask_byte2long__mutmut_mutants: ClassVar[MutantDict] = {
    "x__mask_byte2long__mutmut_1": x__mask_byte2long__mutmut_1,
    "x__mask_byte2long__mutmut_2": x__mask_byte2long__mutmut_2,
    "x__mask_byte2long__mutmut_3": x__mask_byte2long__mutmut_3,
    "x__mask_byte2long__mutmut_4": x__mask_byte2long__mutmut_4,
    "x__mask_byte2long__mutmut_5": x__mask_byte2long__mutmut_5,
    "x__mask_byte2long__mutmut_6": x__mask_byte2long__mutmut_6,
    "x__mask_byte2long__mutmut_7": x__mask_byte2long__mutmut_7,
    "x__mask_byte2long__mutmut_8": x__mask_byte2long__mutmut_8,
    "x__mask_byte2long__mutmut_9": x__mask_byte2long__mutmut_9,
    "x__mask_byte2long__mutmut_10": x__mask_byte2long__mutmut_10,
    "x__mask_byte2long__mutmut_11": x__mask_byte2long__mutmut_11,
    "x__mask_byte2long__mutmut_12": x__mask_byte2long__mutmut_12,
    "x__mask_byte2long__mutmut_13": x__mask_byte2long__mutmut_13,
    "x__mask_byte2long__mutmut_14": x__mask_byte2long__mutmut_14,
    "x__mask_byte2long__mutmut_15": x__mask_byte2long__mutmut_15,
    "x__mask_byte2long__mutmut_16": x__mask_byte2long__mutmut_16,
    "x__mask_byte2long__mutmut_17": x__mask_byte2long__mutmut_17,
    "x__mask_byte2long__mutmut_18": x__mask_byte2long__mutmut_18,
    "x__mask_byte2long__mutmut_19": x__mask_byte2long__mutmut_19,
    "x__mask_byte2long__mutmut_20": x__mask_byte2long__mutmut_20,
    "x__mask_byte2long__mutmut_21": x__mask_byte2long__mutmut_21,
    "x__mask_byte2long__mutmut_22": x__mask_byte2long__mutmut_22,
    "x__mask_byte2long__mutmut_23": x__mask_byte2long__mutmut_23,
    "x__mask_byte2long__mutmut_24": x__mask_byte2long__mutmut_24,
}


def _mask_byte2long(*args, **kwargs):
    result = _mutmut_trampoline(x__mask_byte2long__mutmut_orig, x__mask_byte2long__mutmut_mutants, args, kwargs)
    return result


_mask_byte2long.__signature__ = _mutmut_signature(x__mask_byte2long__mutmut_orig)
x__mask_byte2long__mutmut_orig.__name__ = "x__mask_byte2long"


def x_mask_bool2long__mutmut_orig(mask):
    """
    Converts boolean mask directly to long integers representation.

    Args:
        mask (Tensor): Boolean mask.

    Returns:
        Tensor: Long mask.
    """
    assert mask.dtype == torch.uint8
    return _mask_byte2long(_mask_bool2byte(mask))


def x_mask_bool2long__mutmut_1(mask):
    """
    Converts boolean mask directly to long integers representation.

    Args:
        mask (Tensor): Boolean mask.

    Returns:
        Tensor: Long mask.
    """
    assert mask.dtype != torch.uint8
    return _mask_byte2long(_mask_bool2byte(mask))


def x_mask_bool2long__mutmut_2(mask):
    """
    Converts boolean mask directly to long integers representation.

    Args:
        mask (Tensor): Boolean mask.

    Returns:
        Tensor: Long mask.
    """
    assert mask.dtype == torch.uint8
    return _mask_byte2long(None)


def x_mask_bool2long__mutmut_3(mask):
    """
    Converts boolean mask directly to long integers representation.

    Args:
        mask (Tensor): Boolean mask.

    Returns:
        Tensor: Long mask.
    """
    assert mask.dtype == torch.uint8
    return _mask_byte2long(_mask_bool2byte(None))


x_mask_bool2long__mutmut_mutants: ClassVar[MutantDict] = {
    "x_mask_bool2long__mutmut_1": x_mask_bool2long__mutmut_1,
    "x_mask_bool2long__mutmut_2": x_mask_bool2long__mutmut_2,
    "x_mask_bool2long__mutmut_3": x_mask_bool2long__mutmut_3,
}


def mask_bool2long(*args, **kwargs):
    result = _mutmut_trampoline(x_mask_bool2long__mutmut_orig, x_mask_bool2long__mutmut_mutants, args, kwargs)
    return result


mask_bool2long.__signature__ = _mutmut_signature(x_mask_bool2long__mutmut_orig)
x_mask_bool2long__mutmut_orig.__name__ = "x_mask_bool2long"


def x__mask_long2byte__mutmut_orig(mask, n=None):
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


def x__mask_long2byte__mutmut_1(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is not None:
        n = 8 * mask.size(-1)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_2(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is None:
        n = None
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_3(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is None:
        n = 8 / mask.size(-1)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_4(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is None:
        n = 9 * mask.size(-1)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_5(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is None:
        n = 8 * mask.size(None)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_6(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is None:
        n = 8 * mask.size(+1)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_7(mask, n=None):
    """
    Converts long representation back to byte representation.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Byte mask.
    """
    if n is None:
        n = 8 * mask.size(-2)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_8(mask, n=None):
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
        .view(*mask.size()[:-1], None)[..., :n]
    )


def x__mask_long2byte__mutmut_9(mask, n=None):
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
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.uint8).view(-1)[..., :n]


def x__mask_long2byte__mutmut_10(mask, n=None):
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
        .view(
            *mask.size()[:-1],
        )[..., :n]
    )


def x__mask_long2byte__mutmut_11(mask, n=None):
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
        .to(None)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_12(mask, n=None):
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
        (mask[..., None] << (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_13(mask, n=None):
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
        (mask[..., None] >> (torch.arange(8, out=mask.new()) / 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_14(mask, n=None):
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
        (mask[..., None] >> (torch.arange(None, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_15(mask, n=None):
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
        (mask[..., None] >> (torch.arange(8, out=None) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_16(mask, n=None):
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
        (mask[..., None] >> (torch.arange(out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_17(mask, n=None):
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
        (
            mask[..., None]
            >> (
                torch.arange(
                    8,
                )
                * 8
            )
        )[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_18(mask, n=None):
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
        (mask[..., None] >> (torch.arange(9, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_19(mask, n=None):
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
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 9))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_20(mask, n=None):
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
        .view(*mask.size()[:+1], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_21(mask, n=None):
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
        .view(*mask.size()[:-2], -1)[..., :n]
    )


def x__mask_long2byte__mutmut_22(mask, n=None):
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
        .view(*mask.size()[:-1], +1)[..., :n]
    )


def x__mask_long2byte__mutmut_23(mask, n=None):
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
        .view(*mask.size()[:-1], -2)[..., :n]
    )


x__mask_long2byte__mutmut_mutants: ClassVar[MutantDict] = {
    "x__mask_long2byte__mutmut_1": x__mask_long2byte__mutmut_1,
    "x__mask_long2byte__mutmut_2": x__mask_long2byte__mutmut_2,
    "x__mask_long2byte__mutmut_3": x__mask_long2byte__mutmut_3,
    "x__mask_long2byte__mutmut_4": x__mask_long2byte__mutmut_4,
    "x__mask_long2byte__mutmut_5": x__mask_long2byte__mutmut_5,
    "x__mask_long2byte__mutmut_6": x__mask_long2byte__mutmut_6,
    "x__mask_long2byte__mutmut_7": x__mask_long2byte__mutmut_7,
    "x__mask_long2byte__mutmut_8": x__mask_long2byte__mutmut_8,
    "x__mask_long2byte__mutmut_9": x__mask_long2byte__mutmut_9,
    "x__mask_long2byte__mutmut_10": x__mask_long2byte__mutmut_10,
    "x__mask_long2byte__mutmut_11": x__mask_long2byte__mutmut_11,
    "x__mask_long2byte__mutmut_12": x__mask_long2byte__mutmut_12,
    "x__mask_long2byte__mutmut_13": x__mask_long2byte__mutmut_13,
    "x__mask_long2byte__mutmut_14": x__mask_long2byte__mutmut_14,
    "x__mask_long2byte__mutmut_15": x__mask_long2byte__mutmut_15,
    "x__mask_long2byte__mutmut_16": x__mask_long2byte__mutmut_16,
    "x__mask_long2byte__mutmut_17": x__mask_long2byte__mutmut_17,
    "x__mask_long2byte__mutmut_18": x__mask_long2byte__mutmut_18,
    "x__mask_long2byte__mutmut_19": x__mask_long2byte__mutmut_19,
    "x__mask_long2byte__mutmut_20": x__mask_long2byte__mutmut_20,
    "x__mask_long2byte__mutmut_21": x__mask_long2byte__mutmut_21,
    "x__mask_long2byte__mutmut_22": x__mask_long2byte__mutmut_22,
    "x__mask_long2byte__mutmut_23": x__mask_long2byte__mutmut_23,
}


def _mask_long2byte(*args, **kwargs):
    result = _mutmut_trampoline(x__mask_long2byte__mutmut_orig, x__mask_long2byte__mutmut_mutants, args, kwargs)
    return result


_mask_long2byte.__signature__ = _mutmut_signature(x__mask_long2byte__mutmut_orig)
x__mask_long2byte__mutmut_orig.__name__ = "x__mask_long2byte"


def x__mask_byte2bool__mutmut_orig(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_1(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is not None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_2(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is None:
        n = None
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_3(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is None:
        n = 8 / mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_4(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is None:
        n = 9 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_5(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is None:
        n = 8 * mask.size(None)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_6(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is None:
        n = 8 * mask.size(+1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_7(mask, n=None):
    """
    Converts byte representation back to boolean mask.

    Args:
        mask (Tensor): Byte mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    if n is None:
        n = 8 * mask.size(-2)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_8(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], None)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_9(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(-1)[..., :n] > 0


def x__mask_byte2bool__mutmut_10(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(
        *mask.size()[:-1],
    )[..., :n] > 0


def x__mask_byte2bool__mutmut_11(mask, n=None):
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
    return (mask[..., None] | (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_12(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) >> torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_13(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(None) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_14(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(9) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_15(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) / 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_16(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(None, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_17(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=None) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_18(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_19(mask, n=None):
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
        mask[..., None]
        & (
            mask.new_ones(8)
            << torch.arange(
                8,
            )
            * 1
        )
    ).view(*mask.size()[:-1], -1)[..., :n] > 0


def x__mask_byte2bool__mutmut_20(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(9, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_21(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 2)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_22(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:+1], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_23(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-2], -1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_24(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], +1)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_25(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -2)[
        ..., :n
    ] > 0


def x__mask_byte2bool__mutmut_26(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] >= 0


def x__mask_byte2bool__mutmut_27(mask, n=None):
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
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[
        ..., :n
    ] > 1


x__mask_byte2bool__mutmut_mutants: ClassVar[MutantDict] = {
    "x__mask_byte2bool__mutmut_1": x__mask_byte2bool__mutmut_1,
    "x__mask_byte2bool__mutmut_2": x__mask_byte2bool__mutmut_2,
    "x__mask_byte2bool__mutmut_3": x__mask_byte2bool__mutmut_3,
    "x__mask_byte2bool__mutmut_4": x__mask_byte2bool__mutmut_4,
    "x__mask_byte2bool__mutmut_5": x__mask_byte2bool__mutmut_5,
    "x__mask_byte2bool__mutmut_6": x__mask_byte2bool__mutmut_6,
    "x__mask_byte2bool__mutmut_7": x__mask_byte2bool__mutmut_7,
    "x__mask_byte2bool__mutmut_8": x__mask_byte2bool__mutmut_8,
    "x__mask_byte2bool__mutmut_9": x__mask_byte2bool__mutmut_9,
    "x__mask_byte2bool__mutmut_10": x__mask_byte2bool__mutmut_10,
    "x__mask_byte2bool__mutmut_11": x__mask_byte2bool__mutmut_11,
    "x__mask_byte2bool__mutmut_12": x__mask_byte2bool__mutmut_12,
    "x__mask_byte2bool__mutmut_13": x__mask_byte2bool__mutmut_13,
    "x__mask_byte2bool__mutmut_14": x__mask_byte2bool__mutmut_14,
    "x__mask_byte2bool__mutmut_15": x__mask_byte2bool__mutmut_15,
    "x__mask_byte2bool__mutmut_16": x__mask_byte2bool__mutmut_16,
    "x__mask_byte2bool__mutmut_17": x__mask_byte2bool__mutmut_17,
    "x__mask_byte2bool__mutmut_18": x__mask_byte2bool__mutmut_18,
    "x__mask_byte2bool__mutmut_19": x__mask_byte2bool__mutmut_19,
    "x__mask_byte2bool__mutmut_20": x__mask_byte2bool__mutmut_20,
    "x__mask_byte2bool__mutmut_21": x__mask_byte2bool__mutmut_21,
    "x__mask_byte2bool__mutmut_22": x__mask_byte2bool__mutmut_22,
    "x__mask_byte2bool__mutmut_23": x__mask_byte2bool__mutmut_23,
    "x__mask_byte2bool__mutmut_24": x__mask_byte2bool__mutmut_24,
    "x__mask_byte2bool__mutmut_25": x__mask_byte2bool__mutmut_25,
    "x__mask_byte2bool__mutmut_26": x__mask_byte2bool__mutmut_26,
    "x__mask_byte2bool__mutmut_27": x__mask_byte2bool__mutmut_27,
}


def _mask_byte2bool(*args, **kwargs):
    result = _mutmut_trampoline(x__mask_byte2bool__mutmut_orig, x__mask_byte2bool__mutmut_mutants, args, kwargs)
    return result


_mask_byte2bool.__signature__ = _mutmut_signature(x__mask_byte2bool__mutmut_orig)
x__mask_byte2bool__mutmut_orig.__name__ = "x__mask_byte2bool"


def x_mask_long2bool__mutmut_orig(mask, n=None):
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


def x_mask_long2bool__mutmut_1(mask, n=None):
    """
    Converts long representation back to boolean mask.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    assert mask.dtype != torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def x_mask_long2bool__mutmut_2(mask, n=None):
    """
    Converts long representation back to boolean mask.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    assert mask.dtype == torch.int64
    return _mask_byte2bool(None, n=n)


def x_mask_long2bool__mutmut_3(mask, n=None):
    """
    Converts long representation back to boolean mask.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=None)


def x_mask_long2bool__mutmut_4(mask, n=None):
    """
    Converts long representation back to boolean mask.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    assert mask.dtype == torch.int64
    return _mask_byte2bool(n=n)


def x_mask_long2bool__mutmut_5(mask, n=None):
    """
    Converts long representation back to boolean mask.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    assert mask.dtype == torch.int64
    return _mask_byte2bool(
        _mask_long2byte(mask),
    )


def x_mask_long2bool__mutmut_6(mask, n=None):
    """
    Converts long representation back to boolean mask.

    Args:
        mask (Tensor): Long mask.
        n (int, optional): Original size.

    Returns:
        Tensor: Boolean mask.
    """
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(None), n=n)


x_mask_long2bool__mutmut_mutants: ClassVar[MutantDict] = {
    "x_mask_long2bool__mutmut_1": x_mask_long2bool__mutmut_1,
    "x_mask_long2bool__mutmut_2": x_mask_long2bool__mutmut_2,
    "x_mask_long2bool__mutmut_3": x_mask_long2bool__mutmut_3,
    "x_mask_long2bool__mutmut_4": x_mask_long2bool__mutmut_4,
    "x_mask_long2bool__mutmut_5": x_mask_long2bool__mutmut_5,
    "x_mask_long2bool__mutmut_6": x_mask_long2bool__mutmut_6,
}


def mask_long2bool(*args, **kwargs):
    result = _mutmut_trampoline(x_mask_long2bool__mutmut_orig, x_mask_long2bool__mutmut_mutants, args, kwargs)
    return result


mask_long2bool.__signature__ = _mutmut_signature(x_mask_long2bool__mutmut_orig)
x_mask_long2bool__mutmut_orig.__name__ = "x_mask_long2bool"


def x_mask_long_scatter__mutmut_orig(mask, values, check_unset=True):
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


def x_mask_long_scatter__mutmut_1(mask, values, check_unset=False):
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


def x_mask_long_scatter__mutmut_2(mask, values, check_unset=True):
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
    assert mask.size()[:+1] == values.size()
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


def x_mask_long_scatter__mutmut_3(mask, values, check_unset=True):
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
    assert mask.size()[:-2] == values.size()
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


def x_mask_long_scatter__mutmut_4(mask, values, check_unset=True):
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
    assert mask.size()[:-1] != values.size()
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


def x_mask_long_scatter__mutmut_5(mask, values, check_unset=True):
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
    rng = None
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_6(mask, values, check_unset=True):
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
    rng = torch.arange(None, out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_7(mask, values, check_unset=True):
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
    rng = torch.arange(mask.size(-1), out=None)
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_8(mask, values, check_unset=True):
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
    rng = torch.arange(out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_9(mask, values, check_unset=True):
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
    rng = torch.arange(
        mask.size(-1),
    )
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_10(mask, values, check_unset=True):
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
    rng = torch.arange(mask.size(None), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_11(mask, values, check_unset=True):
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
    rng = torch.arange(mask.size(+1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_12(mask, values, check_unset=True):
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
    rng = torch.arange(mask.size(-2), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_13(mask, values, check_unset=True):
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
    values_ = None  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_14(mask, values, check_unset=True):
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
    # rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = None
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_15(mask, values, check_unset=True):
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
    where = (values_ >= (rng * 64)) | (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_16(mask, values, check_unset=True):
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
    where = (values_ > (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_17(mask, values, check_unset=True):
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
    where = (values_ >= (rng / 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_18(mask, values, check_unset=True):
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
    where = (values_ >= (rng * 65)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_19(mask, values, check_unset=True):
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
    where = (values_ >= (rng * 64)) & (values_ <= ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_20(mask, values, check_unset=True):
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
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) / 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_21(mask, values, check_unset=True):
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
    where = (values_ >= (rng * 64)) & (values_ < ((rng - 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_22(mask, values, check_unset=True):
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
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 2) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_23(mask, values, check_unset=True):
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
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 65))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_24(mask, values, check_unset=True):
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
    assert check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any()
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_25(mask, values, check_unset=True):
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
    assert not (check_unset or ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_26(mask, values, check_unset=True):
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
    assert not (check_unset and ((mask | (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_27(mask, values, check_unset=True):
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
    assert not (check_unset and ((mask & (where.long() >> (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_28(mask, values, check_unset=True):
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
    assert not (check_unset and ((mask & (where.long() << (values_ / 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_29(mask, values, check_unset=True):
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
    assert not (check_unset and ((mask & (where.long() << (values_ % 65))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_30(mask, values, check_unset=True):
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
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) >= 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_31(mask, values, check_unset=True):
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
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 1).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_32(mask, values, check_unset=True):
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
    return mask & (where.long() << (values_ % 64))


def x_mask_long_scatter__mutmut_33(mask, values, check_unset=True):
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
    return mask | (where.long() >> (values_ % 64))


def x_mask_long_scatter__mutmut_34(mask, values, check_unset=True):
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
    return mask | (where.long() << (values_ / 64))


def x_mask_long_scatter__mutmut_35(mask, values, check_unset=True):
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
    return mask | (where.long() << (values_ % 65))


x_mask_long_scatter__mutmut_mutants: ClassVar[MutantDict] = {
    "x_mask_long_scatter__mutmut_1": x_mask_long_scatter__mutmut_1,
    "x_mask_long_scatter__mutmut_2": x_mask_long_scatter__mutmut_2,
    "x_mask_long_scatter__mutmut_3": x_mask_long_scatter__mutmut_3,
    "x_mask_long_scatter__mutmut_4": x_mask_long_scatter__mutmut_4,
    "x_mask_long_scatter__mutmut_5": x_mask_long_scatter__mutmut_5,
    "x_mask_long_scatter__mutmut_6": x_mask_long_scatter__mutmut_6,
    "x_mask_long_scatter__mutmut_7": x_mask_long_scatter__mutmut_7,
    "x_mask_long_scatter__mutmut_8": x_mask_long_scatter__mutmut_8,
    "x_mask_long_scatter__mutmut_9": x_mask_long_scatter__mutmut_9,
    "x_mask_long_scatter__mutmut_10": x_mask_long_scatter__mutmut_10,
    "x_mask_long_scatter__mutmut_11": x_mask_long_scatter__mutmut_11,
    "x_mask_long_scatter__mutmut_12": x_mask_long_scatter__mutmut_12,
    "x_mask_long_scatter__mutmut_13": x_mask_long_scatter__mutmut_13,
    "x_mask_long_scatter__mutmut_14": x_mask_long_scatter__mutmut_14,
    "x_mask_long_scatter__mutmut_15": x_mask_long_scatter__mutmut_15,
    "x_mask_long_scatter__mutmut_16": x_mask_long_scatter__mutmut_16,
    "x_mask_long_scatter__mutmut_17": x_mask_long_scatter__mutmut_17,
    "x_mask_long_scatter__mutmut_18": x_mask_long_scatter__mutmut_18,
    "x_mask_long_scatter__mutmut_19": x_mask_long_scatter__mutmut_19,
    "x_mask_long_scatter__mutmut_20": x_mask_long_scatter__mutmut_20,
    "x_mask_long_scatter__mutmut_21": x_mask_long_scatter__mutmut_21,
    "x_mask_long_scatter__mutmut_22": x_mask_long_scatter__mutmut_22,
    "x_mask_long_scatter__mutmut_23": x_mask_long_scatter__mutmut_23,
    "x_mask_long_scatter__mutmut_24": x_mask_long_scatter__mutmut_24,
    "x_mask_long_scatter__mutmut_25": x_mask_long_scatter__mutmut_25,
    "x_mask_long_scatter__mutmut_26": x_mask_long_scatter__mutmut_26,
    "x_mask_long_scatter__mutmut_27": x_mask_long_scatter__mutmut_27,
    "x_mask_long_scatter__mutmut_28": x_mask_long_scatter__mutmut_28,
    "x_mask_long_scatter__mutmut_29": x_mask_long_scatter__mutmut_29,
    "x_mask_long_scatter__mutmut_30": x_mask_long_scatter__mutmut_30,
    "x_mask_long_scatter__mutmut_31": x_mask_long_scatter__mutmut_31,
    "x_mask_long_scatter__mutmut_32": x_mask_long_scatter__mutmut_32,
    "x_mask_long_scatter__mutmut_33": x_mask_long_scatter__mutmut_33,
    "x_mask_long_scatter__mutmut_34": x_mask_long_scatter__mutmut_34,
    "x_mask_long_scatter__mutmut_35": x_mask_long_scatter__mutmut_35,
}


def mask_long_scatter(*args, **kwargs):
    result = _mutmut_trampoline(x_mask_long_scatter__mutmut_orig, x_mask_long_scatter__mutmut_mutants, args, kwargs)
    return result


mask_long_scatter.__signature__ = _mutmut_signature(x_mask_long_scatter__mutmut_orig)
x_mask_long_scatter__mutmut_orig.__name__ = "x_mask_long_scatter"
