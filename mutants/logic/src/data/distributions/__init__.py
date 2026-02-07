"""
Facade for sampling distributions.
"""

from typing import Any, Callable

from .empirical import Empirical as Empirical
from .spatial import (
    Cluster as Cluster,
)
from .spatial import (
    Gaussian_Mixture as Gaussian_Mixture,
)
from .spatial import (
    Mix_Distribution as Mix_Distribution,
)
from .spatial import (
    Mix_Multi_Distributions as Mix_Multi_Distributions,
)
from .spatial import (
    Mixed as Mixed,
)
from .statistical import Gamma as Gamma

# Registry
DISTRIBUTION_REGISTRY: dict[str, Callable[..., Any]] = {
    "uniform": lambda: None,  # Use default torch.rand
    "cluster": Cluster,
    "mixed": Mixed,
    "gaussian_mixture": Gaussian_Mixture,
    "gamma": Gamma,
    "empirical": Empirical,
    "mix_distribution": Mix_Distribution,
    "mix_multi": Mix_Multi_Distributions,
}

__all__ = [
    "Cluster",
    "Mixed",
    "Gaussian_Mixture",
    "Gamma",
    "Empirical",
    "Mix_Distribution",
    "Mix_Multi_Distributions",
    "DISTRIBUTION_REGISTRY",
]
