"""
Facade for sampling distributions.
"""

from typing import Any, Callable

from .empirical import Empirical as Empirical
from .spatial_cluster import Cluster as Cluster
from .spatial_gaussian_mixture import Gaussian_Mixture as Gaussian_Mixture
from .spatial_mix import Mix_Distribution as Mix_Distribution
from .spatial_mix_multi import Mix_Multi_Distributions as Mix_Multi_Distributions
from .spatial_mixed import Mixed as Mixed
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
