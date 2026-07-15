"""Attention ring-buffer bridge for WSmart-Route Studio (§A.2 Option A).

Serialises runtime encoder attention snapshots into ``ATTENTION_VIZ_START:``
log lines recognised by the Tauri Studio app.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Dict, List, Optional

import logic.src.constants as udef
from logic.src.utils.infrastructure.setup_sims import deep_sanitize

ATTENTION_VIZ_MARKER = "ATTENTION_VIZ_START:"


def send_attention_viz_to_gui(
    snapshots: List[Dict[str, Any]],
    phase: str,
    epoch: int,
    step: int,
    log_path: Optional[str],
    lock: Optional[threading.Lock] = None,
) -> None:
    """Write an ``ATTENTION_VIZ_START:`` line to stdout and optional JSONL log."""
    if not snapshots:
        return

    payload = deep_sanitize(
        {
            "phase": phase,
            "epoch": epoch,
            "step": step,
            "snapshots": snapshots,
        }
    )
    json_payload = json.dumps(payload, separators=(",", ":"))
    log_msg = f"{ATTENTION_VIZ_MARKER}{json_payload}"

    print(log_msg, flush=True)

    if not log_path:
        return

    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except OSError as exc:
        print(f"Warning: Failed to write attention viz to log file: {exc}", file=sys.stderr)
    finally:
        if lock is not None:
            lock.release()


def maybe_emit_attention_viz(
    model: Any,
    cfg: Any,
    phase: str = "eval",
    epoch: int = 0,
    step: int = 0,
    log_path: Optional[str] = None,
    lock: Optional[threading.Lock] = None,
) -> bool:
    """Emit buffered attention snapshots when ``tracking.log_attention`` is enabled."""
    tracking = getattr(cfg, "tracking", None)
    if tracking is None or not bool(getattr(tracking, "log_attention", False)):
        return False

    policy = model
    if hasattr(model, "policy"):
        policy = model.policy

    buffer = getattr(policy, "attention_buffer", None)
    if buffer is None:
        return False

    snapshots = buffer.get_snapshots()
    if not snapshots:
        return False

    if log_path is None:
        log_dir = getattr(tracking, "log_dir", "logs")
        log_path = f"{log_dir}/attention_viz.jsonl"

    send_attention_viz_to_gui(snapshots, phase, epoch, step, log_path, lock)
    return True
