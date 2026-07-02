"""
Batch step executor — concrete implementations for all supported step types.

Each step type corresponds to a handler function registered in the
``STEP_HANDLERS`` registry at the bottom of this module.  The ``BatchExecutor``
calls ``run_step(step, job)`` which dispatches to the appropriate handler.

Adding new step types
---------------------
1. Write a handler with the signature::

       def my_step(args: Dict[str, Any], job: BatchJob) -> None:
           ...

2. Register it::

       STEP_HANDLERS["my_step"] = my_step

The handler should raise an exception to signal failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .batch_job import BatchJob
from .batch_step import BatchStep

__all__ = ["run_step", "STEP_HANDLERS"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the WSmart-Route repository root."""
    # This file lives at logic/controllers/manager/batch_step_executor.py
    return Path(__file__).resolve().parents[3]


def _git(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a git command in the project root.

    Args:
        *args: Git sub-command arguments.
        cwd: Working directory (defaults to project root).

    Returns:
        Completed process object.

    Raises:
        subprocess.CalledProcessError: If git returns a non-zero exit code.
    """
    root = cwd or _project_root()
    cmd = ["git", *args]
    print(f"  [git] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)


def _render(template: str, job: BatchJob) -> str:
    """Render a template string against a job's context variables.

    Args:
        template: Python format string.
        job: Source of template variables.

    Returns:
        Rendered string.
    """
    return job.render_template(template)


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------


def _step_gen_dist_matrix(args: Dict[str, Any], job: BatchJob) -> None:
    """Generate a distance matrix for the given area/problem if missing.

    Args:
        args: Must contain ``area`` (str).  Optional: ``problem`` (str,
            default ``"vrpp"``), ``num_loc`` (int, default 0 = auto),
            ``check_exists`` (bool, default True).
        job: Parent job (unused but available for template rendering).
    """
    area: str = _render(str(args.get("area", "")), job)
    problem: str = args.get("problem", "vrpp")
    num_loc: int = args.get("num_loc", 0)
    check_exists: bool = args.get("check_exists", True)

    if not area:
        raise ValueError("gen_dist_matrix step requires 'area' in args")

    root = _project_root()
    script = root / "logic" / "scripts" / "gen_dist_matrix.py"

    cmd = [sys.executable, str(script), "--area", area, "--problem", problem]
    if num_loc:
        cmd += ["--num_loc", str(num_loc)]
    if check_exists:
        cmd.append("--check_exists")

    print(f"  [gen_dist_matrix] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=root, check=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"gen_dist_matrix failed with exit code {result.returncode}")


def _step_git_add(args: Dict[str, Any], job: BatchJob) -> None:
    """Stage files matching a glob pattern.

    Args:
        args: ``pattern`` (str, default ``"."``) — git add target.
        job: Parent job.
    """
    pattern: str = _render(str(args.get("pattern", ".")), job)
    _git("add", pattern)


def _step_git_commit(args: Dict[str, Any], job: BatchJob) -> None:
    """Commit staged files.

    Args:
        args: ``message`` (str, required) — commit message template.
              ``add_pattern`` (str, optional) — if set, stage this glob before
              committing (equivalent to ``git_add`` + ``git_commit``).
              ``allow_empty`` (bool, default False).
        job: Parent job (used for template rendering in ``message``).
    """
    message_template: str = args.get("message", "Batch commit: {name}")
    message: str = _render(message_template, job)

    add_pattern: Optional[str] = args.get("add_pattern")
    if add_pattern:
        _git("add", _render(add_pattern, job))

    git_args = ["commit", "-m", message]
    if args.get("allow_empty"):
        git_args.append("--allow-empty")

    try:
        _git(*git_args)
    except subprocess.CalledProcessError as exc:
        # Treat "nothing to commit" as a non-fatal warning
        stderr = exc.stderr or ""
        if "nothing to commit" in stderr or "nothing added to commit" in stderr:
            print("  [git_commit] Nothing to commit — skipping.")
        else:
            raise


def _step_git_branch(args: Dict[str, Any], job: BatchJob) -> None:
    """Create and/or checkout a git branch.

    Args:
        args: ``branch_name`` (str, required).
              ``create`` (bool, default True) — if True, create the branch if
              it does not already exist.
        job: Parent job.
    """
    branch: str = _render(str(args.get("branch_name", "")), job)
    if not branch:
        raise ValueError("git_branch step requires 'branch_name' in args")

    create: bool = args.get("create", True)
    if create:
        # Try creating; if it already exists, just checkout
        try:
            _git("checkout", "-b", branch)
        except subprocess.CalledProcessError:
            _git("checkout", branch)
    else:
        _git("checkout", branch)


def _step_git_push(args: Dict[str, Any], job: BatchJob) -> None:
    """Push the current branch to a remote.

    Args:
        args: ``remote`` (str, default ``"origin"``).
              ``branch`` (str, default current branch via ``HEAD``).
              ``set_upstream`` (bool, default True).
        job: Parent job.
    """
    remote: str = args.get("remote", "origin")
    branch: str = _render(str(args.get("branch", "")), job)
    set_upstream: bool = args.get("set_upstream", True)

    git_args = ["push", remote]
    if branch:
        git_args.append(branch)
    if set_upstream:
        git_args += ["--set-upstream", remote, branch or "HEAD"]

    _git(*git_args)


def _step_create_pr(args: Dict[str, Any], job: BatchJob) -> None:
    """Open a pull-request using the ``gh`` CLI.

    Requires ``gh`` to be installed and authenticated.

    Args:
        args: ``title`` (str), ``body`` (str, optional),
              ``base`` (str, default ``"main"``),
              ``draft`` (bool, default False).
        job: Parent job.
    """
    title: str = _render(str(args.get("title", "Batch run results")), job)
    body: str = _render(str(args.get("body", "")), job)
    base: str = args.get("base", "main")
    draft: bool = args.get("draft", False)

    root = _project_root()
    cmd = ["gh", "pr", "create", "--title", title, "--base", base]
    if body:
        cmd += ["--body", body]
    if draft:
        cmd.append("--draft")

    print(f"  [create_pr] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=root, check=True)


def _step_delete_path(args: Dict[str, Any], job: BatchJob) -> None:
    """Delete a file or directory relative to the project root.

    Args:
        args: ``path`` (str, required).
              ``missing_ok`` (bool, default True).
        job: Parent job.
    """
    import shutil

    rel_path: str = _render(str(args.get("path", "")), job)
    if not rel_path:
        raise ValueError("delete_path step requires 'path' in args")

    target = _project_root() / rel_path
    missing_ok: bool = args.get("missing_ok", True)

    if not target.exists():
        if missing_ok:
            print(f"  [delete_path] Path not found (skipping): {rel_path}")
            return
        raise FileNotFoundError(f"delete_path: path not found: {target}")

    if target.is_dir():
        shutil.rmtree(target)
        print(f"  [delete_path] Removed directory: {rel_path}")
    else:
        target.unlink()
        print(f"  [delete_path] Removed file: {rel_path}")


def _step_shell(args: Dict[str, Any], job: BatchJob) -> None:
    """Execute an arbitrary shell command.

    Args:
        args: ``command`` (str, required) — shell command template.
              ``cwd`` (str, optional) — working directory relative to root.
              ``env`` (dict, optional) — extra environment variables.
        job: Parent job.
    """
    command_template: str = args.get("command", "")
    if not command_template:
        raise ValueError("shell step requires 'command' in args")

    command: str = _render(command_template, job)
    cwd_rel: Optional[str] = args.get("cwd")
    root = _project_root()
    cwd = (root / cwd_rel) if cwd_rel else root

    env = None
    if args.get("env"):
        env = {**os.environ, **{k: str(v) for k, v in args["env"].items()}}

    print(f"  [shell] {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"shell step exited with code {result.returncode}: {command}")


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

STEP_HANDLERS: Dict[str, Callable[[Dict[str, Any], BatchJob], None]] = {
    "gen_dist_matrix": _step_gen_dist_matrix,
    "git_add": _step_git_add,
    "git_commit": _step_git_commit,
    "git_branch": _step_git_branch,
    "git_push": _step_git_push,
    "create_pr": _step_create_pr,
    "delete_path": _step_delete_path,
    "shell": _step_shell,
}


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------


def run_step(step: BatchStep, job: BatchJob) -> None:
    """Execute a single batch step.

    Args:
        step: The step to execute.
        job: The current job (provides template context).

    Raises:
        ValueError: If the step type is not registered.
        Any exception raised by the handler itself.
    """
    handler = STEP_HANDLERS.get(step.type)
    if handler is None:
        raise ValueError(
            f"Unknown batch step type: '{step.type}'. "
            f"Available types: {sorted(STEP_HANDLERS)}"
        )
    print(f"  → Running step [{step.display_name()}]")
    handler(step.args, job)
