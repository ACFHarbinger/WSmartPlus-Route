"""
Configuration constants and global mappings.

Attributes:
    dashboard: Dashboard configuration constants
    data: Data configuration constants
    hpo: Hyperparameter optimization configuration constants
    models: Model configuration constants
    paths: Path configuration constants
    routing: Routing configuration constants
    simulation: Simulation configuration constants
    stats: Statistical function mappings
    system: System configuration constants
    testing: Testing configuration constants
    user_interface: User interface configuration constants
    waste: Waste configuration constants

Example:
    >>> from logic.src.constants import *
    >>> ROOT_DIR
    PosixPath('/home/pkhunter/Repositories/WSmart-Route')
"""

from __future__ import annotations

# Re-exporting from split modules
from logic.src.constants.dashboard import *  # noqa: F403
from logic.src.constants.data import *  # noqa: F403
from logic.src.constants.hpo import *  # noqa: F403
from logic.src.constants.models import *  # noqa: F403
from logic.src.constants.paths import *  # noqa: F403
from logic.src.constants.routing import *  # noqa: F403
from logic.src.constants.simulation import *  # noqa: F403
from logic.src.constants.stats import *  # noqa: F403
from logic.src.constants.system import *  # noqa: F403
from logic.src.constants.testing import *  # noqa: F403
from logic.src.constants.user_interface import *  # noqa: F403
from logic.src.constants.waste import *  # noqa: F403
