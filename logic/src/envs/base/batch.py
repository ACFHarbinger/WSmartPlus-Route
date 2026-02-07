"""
Batch handling mixin for RL4CO environment.
"""

import torch


class BatchMixin:
    """
    Mixin to handle batch size management and TorchRL compatibility.
    """

    @property
    def batch_size(self) -> torch.Size:
        """Batch size of the environment."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        """Set the batch size of the environment."""
        # Check if value is a torch.Size object
        if not isinstance(value, torch.Size):
            if isinstance(value, int):
                value = torch.Size([value])
            else:
                value = torch.Size(value)

        try:
            # Try to let EnvBase handle it
            # We assume self is also an instance of EnvBase
            # This is a bit tricky with Mixins calling super() of a sibling.
            # But here we delegate to EnvBase explicitly if we can, or just set it.
            # Actually, standard MRO should handle super().batch_size if EnvBase is in MRO.
            # But EnvBase is the parent of the final class.
            # Mixin -> EnvBase is not strict inheritance, but Mixin is mixed INTO a class inheriting EnvBase.
            # super() in Mixin works if the MRO is correct.
            super(BatchMixin, self.__class__).batch_size.fset(self, value)
        except (ValueError, RuntimeError, AttributeError):
            # Suppress spec re-indexing errors in 0.3.1 or if super fails
            self._batch_size = value

            def _safe_set_shape(s, shp):
                if s is None:
                    return
                try:
                    s.shape = shp
                except (ValueError, RuntimeError):
                    if hasattr(s, "items"):
                        for _, v in s.items():
                            _safe_set_shape(v, shp)

            # Manually sync spec shapes if EnvBase failed to do so
            if hasattr(self, "reward_spec"):
                _safe_set_shape(self.reward_spec, (*value, 1))
            if hasattr(self, "done_spec"):
                _safe_set_shape(self.done_spec, (*value, 1))

            # Sync container shapes
            for spec_name in ["observation_spec", "action_spec", "input_spec", "output_spec"]:
                try:
                    # Use getattr safely as these are often properties
                    spec = getattr(self, spec_name, None)
                    if spec is not None:
                        if hasattr(spec, "shape"):
                            try:
                                spec.shape = value
                            except (ValueError, RuntimeError):
                                pass
                        # Also sync internal done/terminated if they are in there
                        if hasattr(spec, "items"):
                            for k, v in spec.items():
                                if k in ["done", "terminated", "truncated", "reward"]:
                                    _safe_set_shape(v, (*value, 1))
                except (KeyError, AttributeError):
                    continue
        except Exception:
            # Fallback for older TorchRL
            self._batch_size = value
