"""
Batch task configuration dataclass for the WSmart-Route BatchManager.

This module integrates the batch runner into the Hydra config system so that
``BatchConfig`` can be registered in the ``ConfigStore`` and users can pass
``task=batch`` on the command line.

Typical use
-----------
The batch task is **not** routed through ``hydra_dispatch.py``; instead the
``batch_runner.py`` entry-point script calls ``BatchManager.from_yaml()``
directly.  This dataclass is provided so that the batch configuration can be
validated, documented, and potentially composed with other Hydra configs in
the future.

Attributes
----------
``BatchRunConfig``
    Config for a single run entry inside the batch.
``BatchStepConfig``
    Config for a single hook step.
``BatchConfig``
    Root dataclass for the entire batch YAML.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["BatchConfig", "BatchRunConfig", "BatchStepConfig"]


@dataclass
class BatchStepConfig:
    """Configuration for a single lifecycle hook step.

    Attributes:
        type: Step handler key (e.g. ``gen_dist_matrix``, ``git_commit``).
        args: Keyword arguments forwarded to the step handler.
        condition: Execution condition (``always``, ``all_succeeded``,
            ``any_failed``, ``last_succeeded``, ``last_failed``).
        name: Optional human-readable label.
    """

    type: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    condition: str = "always"
    name: Optional[str] = None


@dataclass
class BatchRunConfig:
    """Configuration for a single run entry in a batch.

    Attributes:
        task: Hydra task key (``train``, ``test_sim``, ``eval``, etc.).
        name: Human-readable name for the run.
        overrides: List of Hydra CLI overrides (only for concrete runs).
        pre_steps: Steps to run before the task.
        post_steps: Steps to run after the task.
        expand: Dimension-value mapping for Cartesian-product expansion.
        base_overrides: Base overrides applied to all expanded jobs.
        name_template: Python format string for derived job names.
        dim_overrides: Override built-in dimension → Hydra-override templates.
        metadata: Arbitrary key/value annotations.
    """

    task: str = "test_sim"
    name: str = "run"
    overrides: List[str] = field(default_factory=list)
    pre_steps: List[BatchStepConfig] = field(default_factory=list)
    post_steps: List[BatchStepConfig] = field(default_factory=list)
    expand: Dict[str, List[Any]] = field(default_factory=dict)
    base_overrides: List[str] = field(default_factory=list)
    name_template: Optional[str] = None
    dim_overrides: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Root configuration for a WSmart-Route batch run.

    Attributes:
        name: Human-readable name for the batch (used in reports).
        fail_fast: Abort on the first job failure (default False).
        dry_run: Print commands without executing (default False).
        setup: Steps executed once before all jobs.
        teardown: Steps executed once after all jobs.
        runs: Ordered list of run entries (concrete or expand blocks).
    """

    name: str = "batch"
    fail_fast: bool = False
    dry_run: bool = False
    setup: List[BatchStepConfig] = field(default_factory=list)
    teardown: List[BatchStepConfig] = field(default_factory=list)
    runs: List[BatchRunConfig] = field(default_factory=list)
