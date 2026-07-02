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

Parallel mode
-------------
When ``max_cores > 0`` is passed to ``run_jobs``, the executor launches jobs
concurrently using threads, subject to the per-job ``cores_estimate`` budget:

* A new job is launched as soon as ``used_cores + job.cores_estimate <= max_cores``
  (or immediately if no other jobs are running, to prevent deadlock).
* All git operations (git_add, git_commit, git_branch, git_push, create_pr)
  are serialised with a shared lock so concurrent post-steps cannot corrupt the
  git index simultaneously.
* Each job's Hydra subprocess receives ``job.runtime_env`` merged into its
  environment, enabling per-job config isolation (e.g. ``WSR_POLICY_CONFIG_DIR``).
* ``sim.cpu_cores`` is overridden per-job with ``job.cores_estimate`` so the
  simulation uses exactly the cores allocated to it.

Design rationale
----------------
Running Hydra in a subprocess avoids the singleton ``ConfigStore`` conflict that
arises when re-calling ``hydra.main`` multiple times inside the same process.
Each job therefore starts with a fresh Python interpreter, mirroring how users
would run ``python main.py`` repeatedly from a shell.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .batch_job import BatchJob, JobResult
from .batch_step import BatchStep
from .batch_step_executor import run_step

__all__ = ["BatchExecutor"]

_GIT_STEP_TYPES = frozenset({"git_add", "git_commit", "git_branch", "git_push", "create_pr"})

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
    """Execute a sequence of batch jobs, optionally in parallel.

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
            project_root = Path(__file__).resolve().parents[3]
        self._root = project_root
        self._fail_fast = fail_fast
        self._dry_run = dry_run
        # Set to a Lock in parallel mode; None in sequential mode.
        self._git_lock: Optional[threading.Lock] = None
        self._parallel_mode: bool = False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_jobs(self, jobs: List[BatchJob], max_cores: int = 0) -> List[JobResult]:
        """Execute all jobs, returning their results.

        When ``max_cores > 0`` the executor uses a parallel scheduler.
        Sequential mode (``max_cores <= 0``) retains the original behaviour.

        Args:
            jobs: Ordered list of ``BatchJob`` instances to execute.
            max_cores: Total CPU budget.  0 or negative → sequential.

        Returns:
            List of ``JobResult`` objects, one per job (same order as input).
        """
        if max_cores > 0:
            return self._run_jobs_parallel(jobs, max_cores)
        return self._run_jobs_sequential(jobs)

    # -----------------------------------------------------------------------
    # Sequential execution (original behaviour)
    # -----------------------------------------------------------------------

    def _run_jobs_sequential(self, jobs: List[BatchJob]) -> List[JobResult]:
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
    # Parallel execution
    # -----------------------------------------------------------------------

    def _run_jobs_parallel(self, jobs: List[BatchJob], max_cores: int) -> List[JobResult]:
        """Launch jobs concurrently subject to the ``max_cores`` budget."""
        self._parallel_mode = True
        self._git_lock = threading.Lock()

        results: List[Optional[JobResult]] = [None] * len(jobs)
        cond = threading.Condition()
        used_cores: List[int] = [0]

        def _thread_body(job: BatchJob, idx: int, cores: int) -> None:
            _header(f"[{job.index + 1}/{len(jobs)}] {job.task.upper()} — {job.name} (parallel)")
            result = self._run_single_job(job, [])
            results[idx] = result
            if result.succeeded:
                _ok(f"Job '{job.name}' finished successfully")
            else:
                _fail(f"Job '{job.name}' FAILED")
            with cond:
                used_cores[0] -= cores
                cond.notify_all()

        threads: List[threading.Thread] = []
        for idx, job in enumerate(jobs):
            cores = job.cores_estimate

            with cond:
                # Wait until there is room, but never deadlock: if nothing is
                # running we always proceed regardless of the core count.
                while used_cores[0] > 0 and used_cores[0] + cores > max_cores:
                    cond.wait()
                used_cores[0] += cores

            t = threading.Thread(target=_thread_body, args=(job, idx, cores), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self._parallel_mode = False
        self._git_lock = None
        return [r for r in results if r is not None]

    # -----------------------------------------------------------------------
    # Single-job execution (shared by both modes)
    # -----------------------------------------------------------------------

    def _run_single_job(self, job: BatchJob, succeeded_so_far: List[bool]) -> JobResult:
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

        In parallel mode ``sim.cpu_cores`` is replaced with the job's
        ``cores_estimate`` so each simulation uses exactly its allocated slots.
        ``job.runtime_env`` is merged into the subprocess environment to pass
        per-job config overrides (e.g. ``WSR_POLICY_CONFIG_DIR``).
        """
        main_py = self._root / "main.py"
        overrides = self._effective_overrides(job)
        cmd = [sys.executable, str(main_py), job.task, *overrides]
        print(f"  [task] {' '.join(cmd)}")

        if self._dry_run:
            print("  [dry_run] Skipping actual execution.")
            return 0

        env: Optional[Dict[str, str]] = None
        if job.runtime_env:
            env = {**os.environ, **job.runtime_env}

        proc = subprocess.run(cmd, cwd=self._root, env=env)
        return proc.returncode

    def _effective_overrides(self, job: BatchJob) -> List[str]:
        """Return overrides, substituting sim.cpu_cores in parallel mode."""
        if not self._parallel_mode:
            return job.overrides
        overrides = [o for o in job.overrides if not o.startswith("sim.cpu_cores=")]
        overrides.append(f"sim.cpu_cores={job.cores_estimate}")
        return overrides

    def _execute_step(self, step: BatchStep, job: BatchJob) -> None:
        """Execute a single step, serialising git operations in parallel mode."""
        if self._dry_run:
            print(f"  [dry_run] Would run step: {step.display_name()} args={step.args}")
            return

        if self._git_lock is not None and step.type in _GIT_STEP_TYPES:
            with self._git_lock:
                run_step(step, job)
        else:
            run_step(step, job)
