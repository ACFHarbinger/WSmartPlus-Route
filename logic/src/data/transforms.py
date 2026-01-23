"""
Data augmentation transforms for RL4CO.
"""

import math
from typing import Callable, List, Union

import torch
from tensordict import TensorDict

from logic.src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def batchify(td: TensorDict, num_samples: int) -> TensorDict:
    """Repeats the TensorDict along a new dimension."""
    return td.unsqueeze(1).expand(*td.batch_size, num_samples).reshape(-1)


def dihedral_8_augmentation(xy: torch.Tensor) -> torch.Tensor:
    """
    Augmentation (x8) for grid-based data (x, y) using Dihedral group D4.
    Args:
        xy: [batch, graph, 2] tensor of x and y coordinates
    """
    x, y = xy.split(1, dim=2)
    # Augmentations [batch, graph, 2]
    z0 = torch.cat((x, y), dim=2)
    z1 = torch.cat((1 - x, y), dim=2)
    z2 = torch.cat((x, 1 - y), dim=2)
    z3 = torch.cat((1 - x, 1 - y), dim=2)
    z4 = torch.cat((y, x), dim=2)
    z5 = torch.cat((1 - y, x), dim=2)
    z6 = torch.cat((y, 1 - x), dim=2)
    z7 = torch.cat((1 - y, 1 - x), dim=2)
    # [batch*8, graph, 2]
    aug_xy = torch.cat((z0, z1, z2, z3, z4, z5, z6, z7), dim=0)
    return aug_xy


def dihedral_8_augmentation_wrapper(xy: torch.Tensor, reduce: bool = True, *args, **kw) -> torch.Tensor:
    """Wrapper for dihedral_8_augmentation."""
    if reduce:
        if xy.shape[0] % 8 != 0:
            # Fallback or warning if batch size is not divisible by 8
            log.warning(f"Batch size {xy.shape[0]} is not divisible by 8 for dihedral8 augmentation. Using full batch.")
        else:
            xy = xy[: xy.shape[0] // 8, ...]
    return dihedral_8_augmentation(xy)


def symmetric_transform(x: torch.Tensor, y: torch.Tensor, phi: torch.Tensor, offset: float = 0.5):
    """Symmetric rotation and reflection transform."""
    x, y = x - offset, y - offset
    # Random rotation
    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y
    # Random reflection if phi > 2*pi
    mask = phi > 2 * math.pi
    xy = torch.cat((x_prime, y_prime), dim=-1)
    xy = torch.where(mask, xy.flip(-1), xy)
    return xy + offset


def symmetric_augmentation(xy: torch.Tensor, num_augment: int = 8, first_augment: bool = False):
    """Augment xy data by `num_augment` times via symmetric transform."""
    phi = torch.rand(xy.shape[0], device=xy.device) * 4 * math.pi
    if not first_augment:
        phi[: xy.shape[0] // num_augment] = 0.0
    x, y = xy[..., [0]], xy[..., [1]]
    return symmetric_transform(x, y, phi[:, None, None])


def min_max_normalize(x):
    """
    Normalize tensor to [0, 1] range using min-max scaling.

    Args:
        x: Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    return (x - x.min()) / (x.max() - x.min())


def get_augment_function(augment_fn: Union[str, Callable]):
    """
    Get augmentation function by name or return callable directly.

    Args:
        augment_fn: Either a string ('dihedral8', 'symmetric') or a callable.

    Returns:
        Callable: The augmentation function.

    Raises:
        ValueError: If augment_fn string is not recognized.
    """
    if isinstance(augment_fn, Callable):
        return augment_fn
    if augment_fn == "dihedral8":
        return dihedral_8_augmentation_wrapper
    if augment_fn == "symmetric":
        return symmetric_augmentation
    raise ValueError(f"Unknown augment_fn: {augment_fn}")


class StateAugmentation:
    """
    Augment state by N times via symmetric rotation/reflection transform.
    """

    def __init__(
        self,
        num_augment: int = 8,
        augment_fn: Union[str, Callable] = "symmetric",
        first_aug_identity: bool = True,
        normalize: bool = False,
        feats: List[str] = None,
    ):
        """
        Initialize StateAugmentation transform.

        Args:
            num_augment: Number of augmented copies to generate.
            augment_fn: Augmentation function ('symmetric', 'dihedral8', or callable).
            first_aug_identity: If True, first augmentation is the identity.
            normalize: Whether to apply min-max normalization after augmentation.
            feats: List of feature names to augment (default: ['locs']).
        """
        self.augmentation = get_augment_function(augment_fn)
        if feats is None:
            self.feats = ["locs"]
        else:
            self.feats = feats
        self.num_augment = num_augment
        self.normalize = normalize
        self.first_aug_identity = first_aug_identity

    def __call__(self, td: TensorDict) -> TensorDict:
        """
        Apply augmentation to a TensorDict.

        Args:
            td: Input TensorDict to augment.

        Returns:
            TensorDict: Augmented TensorDict with num_augment copies.
        """
        td_aug = batchify(td, self.num_augment)
        for feat in self.feats:
            if not self.first_aug_identity:
                init_aug_feat = td_aug[feat][list(td.size()), 0].clone()

            aug_feat = self.augmentation(td_aug[feat], num_augment=self.num_augment)

            if self.normalize:
                aug_feat = min_max_normalize(aug_feat)

            if not self.first_aug_identity:
                aug_feat[list(td.size()), 0] = init_aug_feat

            td_aug[feat] = aug_feat
        return td_aug
