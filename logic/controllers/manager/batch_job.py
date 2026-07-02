"""
Batch job definition for the WSmart-Route BatchManager.

A ``BatchJob`` represents a single simulation (or training/evaluation) run with
its associated Hydra overrides, pre-processing steps, and post-processing steps.

Jobs are typically parsed from the ``runs`` list inside a batch YAML file.  They
can also be produced programmatically by ``BatchExpander`` when the YAML uses an
``expand`` block to declare a Cartesian-product of configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .batch_step import BatchStep

__all__ = ["BatchJob", "JobResult"]


@dataclass
class BatchJob:
    """A single executable run within a batch.

    Attributes:
        task: Hydra task key (``train``, ``eval``, ``test_sim``, ``gen_data``,
            ``hpo``, ``meta_train``, ``hpo_sim``).
        name: Human-readable identifier for this job.  Used in log output and
            as the ``{name}`` template variable available to post-steps.
        overrides: List of Hydra CLI override strings applied to this job
            (e.g. ``["sim.policies=[alns,hgs]", "sim.graph.area=figueiradafoz"]``).
        pre_steps: Steps executed immediately before this job.
        post_steps: Steps executed immediately after this job (regardless of
            success/failure unless the step has a condition set).
        index: Zero-based position of this job in the batch sequence.  Set
            automatically by ``BatchManager`` before execution.
        expand_vars: Dictionary of expansion variables used to generate this job
            from an ``expand`` block.  Preserved for template rendering in
            ``BatchStepExecutor``.
        metadata: Arbitrary key/value pairs for user-defined annotations.
    """

    task: str = "test_sim"
    name: str = "run"
    overrides: List[str] = field(default_factory=list)
    pre_steps: List[BatchStep] = field(default_factory=list)
    post_steps: List[BatchStep] = field(default_factory=list)
    index: int = 0
    expand_vars: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Template helpers
    # -----------------------------------------------------------------------

    def render_template(self, template: str) -> str:
        """Render a template string substituting job variables.

        Available template variables:
            ``{name}``      - this job's ``name`` field
            ``{index}``     - zero-based job index
            ``{task}``      - the task key
            plus any keys from ``expand_vars`` flattened as ``{key}``

        Args:
            template: A Python format string.

        Returns:
            The rendered string.
        """
        ctx: Dict[str, Any] = {"name": self.name, "index": self.index, "task": self.task}
        ctx.update(self.expand_vars)
        try:
            return template.format(**ctx)
        except KeyError:
            return template


@dataclass
class JobResult:
    """Outcome of a single batch job execution.

    Attributes:
        job: The job that was executed.
        succeeded: ``True`` if the job and all its post_steps completed without
            raising an exception.
        exit_code: Process exit code returned by Hydra, or -1 on error.
        error: Exception raised during execution (``None`` on success).
        pre_step_errors: Errors raised by individual pre-steps (keyed by step
            display name).
        post_step_errors: Errors raised by individual post-steps.
    """

    job: BatchJob
    succeeded: bool = False
    exit_code: int = -1
    error: Optional[Exception] = None
    pre_step_errors: Dict[str, Exception] = field(default_factory=dict)
    post_step_errors: Dict[str, Exception] = field(default_factory=dict)
