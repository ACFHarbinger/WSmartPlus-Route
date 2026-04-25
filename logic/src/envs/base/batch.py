"""
Batch handling mixin for RL4CO environment.

Attributes:
    BatchMixin: Mixin class for batch size management and TorchRL compatibility.

Example:
    >>> from logic.src.envs.base.batch import BatchMixin
    >>> class MyEnv(BatchMixin):
    ...     def __init__(self):
    ...         self.batch_size = 32
    >>> env = MyEnv()
    >>> env.batch_size
    torch.Size([32])
"""

import contextlib

import torch


class BatchMixin:
    """
    Mixin to handle batch size management and TorchRL compatibility.

    Attributes:
        batch_size: Batch size of the environment.
    """

    @property
    def batch_size(self) -> torch.Size:
        """Batch size of the environment.

        Returns:
            torch.Size: Batch size of the environment.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        """Set the batch size of the environment.

        Args:
            value: Batch size of the environment.
        """
        if not isinstance(value, torch.Size):
            value = torch.Size([value]) if isinstance(value, int) else torch.Size(value)

        try:
            super(BatchMixin, self.__class__).batch_size.fset(self, value)  # type: ignore[misc]
        except (ValueError, RuntimeError, AttributeError):
            self._batch_size = value
            self._sync_spec_shapes(value)

    def _sync_spec_shapes(self, value: torch.Size) -> None:
        """Synchronize spec shapes with the new batch size.

        Args:
            value: Shape to synchronize with.
        """
        if hasattr(self, "reward_spec"):
            self._safe_set_shape(self.reward_spec, (*value, 1))
        if hasattr(self, "done_spec"):
            self._safe_set_shape(self.done_spec, (*value, 1))

        for spec_name in ["observation_spec", "action_spec", "input_spec", "output_spec"]:
            with contextlib.suppress(KeyError, AttributeError):
                spec = getattr(self, spec_name, None)
                if spec is None:
                    continue
                if hasattr(spec, "shape"):
                    with contextlib.suppress(ValueError, RuntimeError):
                        spec.shape = value
                if hasattr(spec, "items"):
                    for k, v in spec.items():
                        if k in ["done", "terminated", "truncated", "reward"]:
                            self._safe_set_shape(v, (*value, 1))

    @staticmethod
    def _safe_set_shape(s, shp):
        """Recursively set shape with error handling.

        Args:
            s: Shape to set.
            shp: New shape.
        """
        if s is None:
            return
        try:
            s.shape = shp
        except (ValueError, RuntimeError):
            if hasattr(s, "items"):
                for v in s.values():
                    BatchMixin._safe_set_shape(v, shp)
