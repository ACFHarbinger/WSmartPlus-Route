"""Tests for TrainingHealthCallback and emit bridge (§A.4)."""

import json
from unittest.mock import MagicMock

import torch

from logic.src.pipeline.callbacks.pytorch.training_health import TrainingHealthCallback
from logic.src.tracking.logging.modules.training_health_emit import (
    TRAINING_HEALTH_MARKER,
    emit_training_health_alert,
)


def test_emit_training_health_alert_stdout(capsys):
    emit_training_health_alert(
        code="grad_norm_explosion",
        severity="critical",
        epoch=2,
        step=50,
        details={"grad_norm": 150.0, "threshold": 100.0},
    )
    captured = capsys.readouterr()
    assert TRAINING_HEALTH_MARKER in captured.out
    payload = json.loads(captured.out.strip().split(TRAINING_HEALTH_MARKER, 1)[1])
    assert payload["code"] == "grad_norm_explosion"
    assert payload["severity"] == "critical"
    assert payload["epoch"] == 2
    assert payload["step"] == 50


def test_emit_training_health_alert_jsonl(tmp_path):
    log_path = str(tmp_path / "training_health.jsonl")
    emit_training_health_alert(
        code="entropy_collapse",
        severity="warning",
        epoch=0,
        step=10,
        details={"entropy": 0.001},
        log_path=log_path,
    )
    content = (tmp_path / "training_health.jsonl").read_text()
    assert TRAINING_HEALTH_MARKER in content
    payload = json.loads(content.strip().split(TRAINING_HEALTH_MARKER, 1)[1])
    assert payload["code"] == "entropy_collapse"


def _make_trainer(epoch: int = 0, step: int = 0, log_dir: str | None = None) -> MagicMock:
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = epoch
    trainer.global_step = step
    trainer.log_dir = log_dir
    trainer.callback_metrics = {}
    return trainer


def test_grad_norm_explosion_alert(capsys):
    cb = TrainingHealthCallback(max_grad_norm_threshold=100.0, alert_cooldown_epochs=0)
    trainer = _make_trainer(step=42)
    trainer.callback_metrics = {"train/grad_norm": torch.tensor(150.0)}

    cb.on_train_batch_end(trainer, MagicMock(), None, None, 0)

    captured = capsys.readouterr()
    assert TRAINING_HEALTH_MARKER in captured.out
    payload = json.loads(captured.out.strip().split(TRAINING_HEALTH_MARKER, 1)[1])
    assert payload["code"] == "grad_norm_explosion"
    assert payload["severity"] == "critical"


def test_entropy_collapse_alert(capsys):
    cb = TrainingHealthCallback(min_entropy_threshold=0.01, alert_cooldown_epochs=0)
    trainer = _make_trainer()
    trainer.callback_metrics = {"train/entropy": torch.tensor(0.005)}

    cb.on_train_batch_end(trainer, MagicMock(), None, None, 0)

    captured = capsys.readouterr()
    assert TRAINING_HEALTH_MARKER in captured.out
    payload = json.loads(captured.out.strip().split(TRAINING_HEALTH_MARKER, 1)[1])
    assert payload["code"] == "entropy_collapse"


def test_reward_stagnation_alert(capsys):
    cb = TrainingHealthCallback(
        stagnation_epochs=3,
        stagnation_epsilon=1e-3,
        alert_cooldown_epochs=0,
    )
    trainer = _make_trainer(epoch=5)
    trainer.callback_metrics = {"train/reward": torch.tensor(-2.5)}

    for _ in range(4):
        cb.on_train_epoch_end(trainer, MagicMock())
        trainer.current_epoch += 1

    captured = capsys.readouterr()
    assert TRAINING_HEALTH_MARKER in captured.out
    payload = json.loads(captured.out.strip().split(TRAINING_HEALTH_MARKER, 1)[1])
    assert payload["code"] == "reward_stagnation"


def test_alert_cooldown_suppresses_repeat(capsys):
    cb = TrainingHealthCallback(max_grad_norm_threshold=10.0, alert_cooldown_epochs=5)
    trainer = _make_trainer(epoch=0, step=1)
    trainer.callback_metrics = {"train/grad_norm": torch.tensor(50.0)}

    cb.on_train_batch_end(trainer, MagicMock(), None, None, 0)
    cb.on_train_batch_end(trainer, MagicMock(), None, None, 1)

    captured = capsys.readouterr()
    assert captured.out.count(TRAINING_HEALTH_MARKER) == 1
