"""
Facade for sampling distributions.
"""

from typing import Any, Callable

from .spatial_distance import Distance as Distance
from .spatial_cluster import Cluster as Cluster
from .spatial_gaussian_mixture import Gaussian_Mixture as Gaussian_Mixture
from .spatial_mix import Mix_Distribution as Mix_Distribution
from .spatial_mix_multi import Mix_Multi_Distributions as Mix_Multi_Distributions
from .spatial_mixed import Mixed as Mixed
from .statistical_empirical import Empirical as Empirical
from .statistical_beta import Beta as Beta
from .statistical_constant import Constant as Constant
from .statistical_gamma import Gamma as Gamma
from .statistical_uniform import Uniform as Uniform

# Registry
DISTRIBUTION_REGISTRY: dict[str, Callable[..., Any]] = {
    "cluster": Cluster,
    "mixed": Mixed,
    "gaussian_mixture": Gaussian_Mixture,
    "gamma": Gamma,
    "empirical": Empirical,
    "mix_distribution": Mix_Distribution,
    "mix_multi": Mix_Multi_Distributions,
    "distance": Distance,
    "beta": Beta,
    "constant": Constant,
    "uniform": Uniform,
}

__all__ = [
    "Cluster",
    "Mixed",
    "Gaussian_Mixture",
    "Gamma",
    "Empirical",
    "Mix_Distribution",
    "Mix_Multi_Distributions",
    "Distance",
    "Beta",
    "Constant",
    "Uniform",
    "DISTRIBUTION_REGISTRY",
]
