"""
Interfaces for WSmart-Route logic components.
DEFINING PROTOCOLS TO DECOUPLE MODULES.
"""

from .adapter import IPolicyAdapter
from .bin_container import IBinContainer
from .env import IEnv
from .model import IModel
from .must_go import IMustGoSelectionStrategy
from .policy import IPolicy
from .post_processing import IPostProcessor
from .tensor_dict_like import ITensorDictLike
from .traversable import ITraversable

__all__ = [
    "IPolicyAdapter",
    "IEnv",
    "IModel",
    "IMustGoSelectionStrategy",
    "IPolicy",
    "IPostProcessor",
    "ITensorDictLike",
    "ITraversable",
    "IBinContainer",
]
