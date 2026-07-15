"""Training health alert bridge for WSmart-Route Studio (§A.4).

Serialises :class:`~logic.src.pipeline.callbacks.pytorch.training_health.TrainingHealthCallback`
warnings into ``TRAINING_HEALTH_START:`` log lines recognised by the Tauri Studio app.

Example:
    >>> emit_training_health_alert("grad_norm_explosion", "warning", 3, 120, {...})
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from logic.src.utils.infrastructure.setup_sims import deep_sanitize

TRAINING_HEALTH_MARKER = "TRAINING_HEALTH_START:"


def emit_training_health_alert(
    code: str,
    severity: str,
    epoch: int,
    step: int,
    details: Optional[Dict[str, Any]] = None,
    log_path: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    """Write a ``TRAINING_HEALTH_START:`` line to stdout and an optional JSONL log.

    Args:
        code: Machine-readable alert code (e.g. ``grad_norm_explosion``).
        severity: ``warning`` or ``critical``.
        epoch: Training epoch index (0-based).
        step: Global training step.
        details: Optional metric snapshot dict.
        log_path: Optional JSONL path for append.
        message: Optional human-readable summary.
    """
    payload: Dict[str, Any] = {
        "code": code,
        "severity": severity,
        "epoch": epoch,
        "step": step,
        "message": message or _default_message(code),
        "details": deep_sanitize(details or {}),
    }
    json_payload = json.dumps(payload, separators=(",", ":"))
    log_msg = f"{TRAINING_HEALTH_MARKER}{json_payload}"

    print(log_msg, flush=True)

    if not log_path:
        return

    try:
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except OSError as exc:
        print(f"Warning: Failed to write training health alert to log file: {exc}", file=sys.stderr)


def _default_message(code: str) -> str:
    messages = {
        "grad_norm_explosion": "Gradient norm exceeded the safety threshold",
        "reward_stagnation": "Reward moving average has stagnated",
        "entropy_collapse": "Policy entropy fell below the collapse threshold",
    }
    return messages.get(code, f"Training health alert: {code}")
