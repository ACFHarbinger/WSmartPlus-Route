"""
Batch executor — runs a sequence of ``BatchJob`` instances produced by the
``BatchManager``.

Execution flow
--------------
For each job in the batch:

1. **Pre-steps**: Run any steps whose ``condition`` evaluates to ``True``
   given the batch progress so far.
2. **Hydra task**: Invoke ``python main.py <task> [overrides...]`` in a
   subprocess so that each run gets a completely clean Hydra state.
3. **Post-steps**: Run any steps whose ``condition`` evaluates to ``True``
   (e.g. ``always``, ``last_succeeded``, etc.).
4. Record the result in a ``JobResult`` and continue (or abort on
   ``fail_fast=True``).

After all jobs:

5. **Setup / teardown steps** are orchestrated by ``BatchManager``, not by
   this class.

Design rationale
----------------
Running Hydra in a subprocess avoids the singleton ``ConfigStore`` conflict that
arises when re-calling ``hydra.main`` multiple times inside the same process.
Each job therefore starts with a fresh Python interpreter, mirroring how users
would run ``python main.py`` repeatedly from a shell.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .batch_job import BatchJob, JobResult
from .batch_step import BatchStep
from .batch_step_executor import run_step

__all__ = ["BatchExecutor"]

_BRIGHT = "\033[1m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"


def _header(text: str, width: int = 70) -> None:
    print(f"\n{_CYAN}{'─' * width}{_RESET}")
    print(f"{_BRIGHT}{text}{_RESET}")
    print(f"{_CYAN}{'─' * width}{_RESET}")


def _ok(text: str) -> None:
    print(f"{_GREEN}✔ {text}{_RESET}")


def _fail(text: str) -> None:
    print(f"{_RED}✘ {text}{_RESET}")


def _warn(text: str) -> None:
    print(f"{_YELLOW}⚠ {text}{_RESET}")


class BatchExecutor:
    """Execute a sequence of batch jobs.

    Parameters
    ----------
    project_root:
        Absolute path to the WSmart-Route repository root.
    fail_fast:
        Abort the batch immediately if any job or pre-step fails.
    dry_run:
        Print what would be executed without actually running anything.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        fail_fast: bool = False,
        dry_run: bool = False,
    ) -> None:
        if project_root is None:
            # This file lives at logic/controllers/manager/batch_executor.py
            project_root = Path(__file__).resolve().parents[3]
        self._root = project_root
        self._fail_fast = fail_fast
        self._dry_run = dry_run

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_jobs(self, jobs: List[BatchJob]) -> List[JobResult]:
        """Execute all jobs in sequence, returning their results.

        Args:
            jobs: Ordered list of ``BatchJob`` instances to execute.

        Returns:
            List of ``JobResult`` objects, one per job.
        """
        results: List[JobResult] = []
        succeeded_flags: List[bool] = []

        for job in jobs:
            _header(f"[{job.index + 1}/{len(jobs)}] {job.task.upper()} — {job.name}")
            result = self._run_single_job(job, succeeded_flags)
            results.append(result)
            succeeded_flags.append(result.succeeded)

            if result.succeeded:
                _ok(f"Job '{job.name}' finished successfully")
            else:
                _fail(f"Job '{job.name}' FAILED")

            if self._fail_fast and not result.succeeded:
                _warn("fail_fast=True — aborting remaining jobs")
                break

        return results

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _run_single_job(self, job: BatchJob, succeeded_so_far: List[bool]) -> JobResult:
        """Execute one job: pre-steps → task → post-steps.

        Args:
            job: The job to execute.
            succeeded_so_far: Boolean list of prior job outcomes.

        Returns:
            ``JobResult`` capturing what happened.
        """
        last_succeeded: Optional[bool] = succeeded_so_far[-1] if succeeded_so_far else None
        result = JobResult(job=job)
        start_time = time.monotonic()

        # --- Pre-steps -------------------------------------------------------
        for step in job.pre_steps:
            if not step.should_run(succeeded=succeeded_so_far, last_succeeded=last_succeeded):
                _warn(f"  Skipping pre-step [{step.display_name()}] (condition not met)")
                continue
            try:
                self._execute_step(step, job)
            except Exception as exc:  # noqa: BLE001
                _fail(f"  Pre-step [{step.display_name()}] failed: {exc}")
                result.pre_step_errors[step.display_name()] = exc
                if self._fail_fast:
                    return result

        # --- Hydra task ------------------------------------------------------
        try:
            exit_code = self._execute_task(job)
            result.exit_code = exit_code
            result.succeeded = exit_code == 0
        except Exception as exc:  # noqa: BLE001
            _fail(f"  Task execution raised: {exc}")
            result.error = exc
            result.exit_code = -1
            result.succeeded = False

        elapsed = time.monotonic() - start_time
        print(f"  Task duration: {elapsed:.1f}s")

        # --- Post-steps -------------------------------------------------------
        # Updated succeeded list includes the just-executed job
        updated_succeeded = list(succeeded_so_far) + [result.succeeded]
        updated_last = result.succeeded

        for step in job.post_steps:
            if not step.should_run(succeeded=updated_succeeded, last_succeeded=updated_last):
                _warn(f"  Skipping post-step [{step.display_name()}] (condition not met)")
                continue
            try:
                self._execute_step(step, job)
            except Exception as exc:  # noqa: BLE001
                _fail(f"  Post-step [{step.display_name()}] failed: {exc}")
                result.post_step_errors[step.display_name()] = exc

        return result

    def _execute_task(self, job: BatchJob) -> int:
        """Launch the Hydra task in a subprocess.

        Args:
            job: Job whose task and overrides to execute.

        Returns:
            Process exit code (0 on success).
        """
        main_py = self._root / "main.py"
        cmd = [sys.executable, str(main_py), job.task, *job.overrides]
        print(f"  [task] {' '.join(cmd)}")

        if self._dry_run:
            print("  [dry_run] Skipping actual execution.")
            return 0

        proc = subprocess.run(cmd, cwd=self._root)
        return proc.returncode

    def _execute_step(self, step: BatchStep, job: BatchJob) -> None:
        """Execute a single step, respecting dry-run mode.

        Args:
            step: Step to run.
            job: Owning job (provides template context).
        """
        if self._dry_run:
            print(f"  [dry_run] Would run step: {step.display_name()} args={step.args}")
            return
        run_step(step, job)
