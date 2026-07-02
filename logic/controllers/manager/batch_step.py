"""
Batch step definitions for the WSmart-Route BatchManager.

A ``BatchStep`` describes a single hook that runs before or after a simulation
execution (pre/post steps) or before/after the entire batch (setup/teardown steps).

Supported step types (``type`` field)
--------------------------------------
- ``gen_dist_matrix``   - Generate a distance matrix for a given area/problem if
                          one does not already exist on disk.
- ``git_add``           - Stage files matching a glob pattern for a git commit.
- ``git_commit``        - Commit currently staged files with a message (supports
                          ``{name}`` and ``{index}`` template variables).
- ``git_branch``        - Create and/or checkout a git branch.
- ``git_push``          - Push the current branch to the remote.
- ``create_pr``         - Open a GitHub/GitLab pull-request via the ``gh`` or
                          ``glab`` CLI (requires CLI to be installed and auth'd).
- ``delete_path``       - Delete a file or directory (use with care).
- ``shell``             - Execute an arbitrary shell command.

All steps support an optional ``condition`` field.  When set, the step is only
executed when the condition evaluates to ``True``.  Available condition values:

- ``"always"``          - (default) Always run.
- ``"all_succeeded"``   - Only run if every previous job finished successfully.
- ``"any_failed"``      - Only run if at least one job failed.
- ``"last_succeeded"``  - Only run if the immediately preceding job succeeded.
- ``"last_failed"``     - Only run if the immediately preceding job failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["BatchStep", "StepCondition"]

# ---------------------------------------------------------------------------
# Condition constants
# ---------------------------------------------------------------------------

CONDITION_ALWAYS = "always"
CONDITION_ALL_SUCCEEDED = "all_succeeded"
CONDITION_ANY_FAILED = "any_failed"
CONDITION_LAST_SUCCEEDED = "last_succeeded"
CONDITION_LAST_FAILED = "last_failed"

StepCondition = str  # type alias for readability


# ---------------------------------------------------------------------------
# BatchStep dataclass
# ---------------------------------------------------------------------------


@dataclass
class BatchStep:
    """A single hook step within a batch run.

    Attributes:
        type: The step handler key (e.g. ``gen_dist_matrix``, ``git_commit``).
        args: Keyword arguments forwarded to the step handler.
        condition: When to execute this step.  See module docstring for valid
            values.  Defaults to ``"always"``.
        name: Optional human-readable label used in log output.
    """

    type: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    condition: StepCondition = CONDITION_ALWAYS
    name: Optional[str] = None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def should_run(
        self,
        *,
        succeeded: List[bool],
        last_succeeded: Optional[bool] = None,
    ) -> bool:
        """Decide whether this step should execute given batch progress.

        Args:
            succeeded: Boolean list tracking success/failure of each job so far.
            last_succeeded: Success status of the immediately preceding job, or
                ``None`` if no jobs have run yet.

        Returns:
            ``True`` if this step should be executed.
        """
        cond = self.condition or CONDITION_ALWAYS
        if cond == CONDITION_ALWAYS:
            return True
        if cond == CONDITION_ALL_SUCCEEDED:
            return bool(succeeded) and all(succeeded)
        if cond == CONDITION_ANY_FAILED:
            return any(not s for s in succeeded)
        if cond == CONDITION_LAST_SUCCEEDED:
            return last_succeeded is True
        if cond == CONDITION_LAST_FAILED:
            return last_succeeded is False
        return True

    def display_name(self) -> str:
        """Return a short human-readable label for this step."""
        return self.name or self.type
