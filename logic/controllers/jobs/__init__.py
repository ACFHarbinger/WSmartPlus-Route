"""Jobs controller package.

Task runners and job execution functions for the WSmart-Route application.

Hydra Pipeline Runner
---------------------
:mod:`~logic.controllers.jobs.pipeline_runner` provides:
- :func:`~logic.controllers.jobs.pipeline_runner.run_training` — train / meta_train / hpo
- :func:`~logic.controllers.jobs.pipeline_runner.run_evaluation` — eval
- :func:`~logic.controllers.jobs.pipeline_runner.run_simulation` — test_sim / hpo_sim
- :func:`~logic.controllers.jobs.pipeline_runner.run_data_generation` — gen_data

Argparse Operations Runner
--------------------------
:mod:`~logic.controllers.jobs.ops_runner` provides:
- :func:`~logic.controllers.jobs.ops_runner.run_benchmarks` — benchmark
- :func:`~logic.controllers.jobs.ops_runner.run_test_suite` — test_suite
- :func:`~logic.controllers.jobs.ops_runner.run_file_system` — file_system
- :func:`~logic.controllers.jobs.ops_runner.run_output_command` — clean_results / excel_summary
- :func:`~logic.controllers.jobs.ops_runner.run_target_update` — update_ms / update_ri
"""

from .ops_runner import (
    run_benchmarks,
    run_file_system,
    run_output_command,
    run_target_update,
    run_test_suite,
)
from .pipeline_runner import (
    run_data_generation,
    run_evaluation,
    run_simulation,
    run_training,
)

__all__ = [
    "run_training",
    "run_evaluation",
    "run_simulation",
    "run_data_generation",
    "run_benchmarks",
    "run_test_suite",
    "run_file_system",
    "run_output_command",
    "run_target_update",
]
