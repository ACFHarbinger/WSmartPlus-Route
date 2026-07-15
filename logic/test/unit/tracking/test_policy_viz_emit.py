"""Tests for policy telemetry emission (§A.3)."""

import json

from logic.src.tracking.logging.modules.policy_viz_emit import (
    POLICY_VIZ_MARKER,
    detect_policy_viz_type,
    maybe_emit_policy_viz,
    send_policy_viz_to_gui,
)
from logic.src.tracking.viz_mixin import PolicyVizMixin


class _DummyPolicy(PolicyVizMixin):
    def run(self) -> None:
        for i in range(3):
            self._viz_record(iteration=i, best_cost=100.0 - i)


def test_detect_policy_viz_type_alns():
    assert detect_policy_viz_type({"d_idx": [0], "best_cost": [1.0]}) == "alns"


def test_detect_policy_viz_type_hgs():
    assert detect_policy_viz_type({"generation": [0], "best_cost": [1.0]}) == "hgs"


def test_detect_policy_viz_type_generic():
    assert detect_policy_viz_type({"iteration": [0], "best_cost": [1.0]}) == "generic"


def test_maybe_emit_policy_viz_from_mixin(capsys, tmp_path):
    policy = _DummyPolicy()
    policy.run()
    log_path = str(tmp_path / "sim.jsonl")

    emitted = maybe_emit_policy_viz(policy, "Test ALNS", 0, 2, log_path)
    assert emitted is True

    captured = capsys.readouterr()
    assert POLICY_VIZ_MARKER in captured.out

    content = (tmp_path / "sim.jsonl").read_text()
    assert POLICY_VIZ_MARKER in content
    payload = content.split(",", 4)[4]
    parsed = json.loads(payload)
    assert parsed["best_cost"] == [100.0, 99.0, 98.0]


def test_send_policy_viz_to_gui_skips_empty(capsys):
    send_policy_viz_to_gui({}, "P", 0, 1, None)
    assert capsys.readouterr().out == ""
