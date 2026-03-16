"""
Facade for sampling distributions.
"""

from typing import Any, Callable

from .base import BaseDistribution
from .spatial_cluster import Cluster as Cluster
from .spatial_distance import Distance as Distance
from .spatial_gaussian_mixture import GaussianMixture as GaussianMixture
from .spatial_mix import MixDistribution as MixDistribution
from .spatial_mix_multi import MixMultiDistributions as MixMultiDistributions
from .spatial_mixed import Mixed as Mixed
from .statistical_beta import Beta as Beta
from .statistical_constant import Constant as Constant
from .statistical_empirical import Empirical as Empirical
from .statistical_gamma import Gamma as Gamma
from .statistical_uniform import Uniform as Uniform

# Registry
DISTRIBUTION_REGISTRY: dict[str, Callable[..., Any]] = {
    "cluster": Cluster,
    "mixed": Mixed,
    "gaussian_mixture": GaussianMixture,
    "gamma": Gamma,
    "empirical": Empirical,
    "mix_distribution": MixDistribution,
    "mix_multi": MixMultiDistributions,
    "distance": Distance,
    "beta": Beta,
    "constant": Constant,
    "uniform": Uniform,
}

__all__ = [
    "BaseDistribution",
    "Cluster",
    "Mixed",
    "Gaussian_Mixture",
    "Gamma",
    "Empirical",
    "MixDistribution",
    "MixMultiDistributions",
    "Distance",
    "Beta",
    "Constant",
    "Uniform",
    "DISTRIBUTION_REGISTRY",
]
