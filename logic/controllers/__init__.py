"""Controller module.

Entry points and command controllers for the WSmart-Route application.

Hydra-driven controllers
------------------------
All neural-model commands (``train``, ``meta_train``, ``hpo``, ``eval``) are
handled by :mod:`~logic.controllers.model_pipeline`.  Simulation and data
commands (``test_sim``, ``hpo_sim``, ``gen_data``) are handled by
:mod:`~logic.controllers.simulation`.

Argparse-driven controllers
----------------------------
Operational commands (``benchmark``, ``test_suite``, ``file_system``,
``clean_results``, ``excel_summary``, ``update_ms``, ``update_ri``) are
handled by :mod:`~logic.controllers.ops`.

Dispatchers
-----------
- :func:`~logic.controllers.hydra_dispatch.hydra_entry_point` — Hydra main
- :func:`~logic.controllers.parser_dispatch.parser_entry_point` — argparse main

Sub-packages
------------
- :mod:`logic.controllers.cli`     — argument parsers and ``parse_params()``
- :mod:`logic.controllers.manager` — ``BatchManager`` for multi-run experiments

Example::

    >>> from logic.controllers import parser_entry_point, hydra_entry_point
    >>> from logic.controllers.cli import parse_params
    >>> parser_entry_point(parse_params())
"""

from .hydra_dispatch import hydra_entry_point
from .parser_dispatch import parser_entry_point

__all__ = [
    "parser_entry_point",
    "hydra_entry_point",
]
