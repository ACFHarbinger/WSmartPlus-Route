"""
Interfaces for WSmart-Route logic components.
DEFINING PROTOCOLS TO DECOUPLE MODULES.
"""

from .adapter import IPolicyAdapter
from .env import IEnv
from .model import IModel
from .must_go import IMustGoSelectionStrategy
from .policy import IPolicy
from .post_processing import IPostProcessor
