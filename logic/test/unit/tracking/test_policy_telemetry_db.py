"""Tests for policy telemetry SQLite persistence (§A.3 Option C)."""

import os

import pytest
from logic.src.tracking.logging.modules import policy_telemetry_db as db
from logic.src.tracking.logging.modules.policy_viz_emit import (
    POLICY_VIZ_MARKER,
    maybe_emit_policy_viz,
)
from logic.src.tracking.viz_mixin import PolicyVizMixin


@pytest.fixture(autouse=True)
def isolated_telemetry_db(tmp_path, monkeypatch):
    test_db = tmp_path / "telemetry.db"
    monkeypatch.setattr(db, "TELEMETRY_DB_PATH", str(test_db))
    yield test_db


class _AlnsPolicy(PolicyVizMixin):
    def run(self) -> None:
        for i in range(3):
            self._viz_record(iteration=i, best_cost=100.0 - i, d_idx=0)


def test_extract_final_metric_alns():
    metric, name = db.extract_final_metric(
        "alns", {"best_cost": [10.0, 9.0, 8.5], "iteration": [0, 1, 2]}
    )
    assert metric == pytest.approx(8.5)
    assert name == "best_cost"


def test_persist_and_query_roundtrip():
    viz = {"iteration": [0, 1], "best_cost": [50.0, 45.0], "d_idx": [0, 1]}
    assert db.persist_policy_viz_snapshot(
        viz, "ALNS + Ftsp", 0, 2, "alns", "/tmp/run_a.jsonl"
    )

    result = db.query_policy_telemetry_trends(policy_type="alns")
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["policy"] == "ALNS + Ftsp"
    assert row["final_metric"] == pytest.approx(45.0)
    assert row["metric_name"] == "best_cost"
    assert row["step_count"] == 2


def test_persist_replaces_stale_snapshot():
    log_path = "/tmp/run_b.jsonl"
    db.persist_policy_viz_snapshot(
        {"iteration": [0], "best_cost": [20.0]}, "P", 0, 1, "alns", log_path
    )
    db.persist_policy_viz_snapshot(
        {"iteration": [0, 1], "best_cost": [20.0, 18.0]},
        "P",
        0,
        1,
        "alns",
        log_path,
    )

    rows = db.query_policy_telemetry_trends()["rows"]
    assert len(rows) == 1
    assert rows[0]["final_metric"] == pytest.approx(18.0)
    assert rows[0]["step_count"] == 2


def test_emit_writes_sqlite(capsys, tmp_path):
    policy = _AlnsPolicy()
    policy.run()
    log_path = str(tmp_path / "sim.jsonl")

    maybe_emit_policy_viz(policy, "Stream ALNS", 0, 1, log_path)
    capsys.readouterr()

    rows = db.query_policy_telemetry_trends()["rows"]
    assert len(rows) == 1
    assert rows[0]["policy"] == "Stream ALNS"
    assert rows[0]["policy_type"] == "alns"

    content = (tmp_path / "sim.jsonl").read_text()
    assert POLICY_VIZ_MARKER in content


def test_query_empty_when_db_missing(monkeypatch):
    missing = "/nonexistent/telemetry.db"
    monkeypatch.setattr(db, "TELEMETRY_DB_PATH", missing)
    if os.path.exists(missing):
        os.remove(missing)
    result = db.query_policy_telemetry_trends()
    assert result["rows"] == []


def test_query_trajectory_series_roundtrip():
    viz = {
        "iteration": [0, 1, 2, 3],
        "best_cost": [50.0, 45.0, 42.0, 40.0],
        "d_idx": [0, 1, 2, 3],
    }
    db.persist_policy_viz_snapshot(
        viz, "ALNS + Ftsp", 0, 2, "alns", "/tmp/run_traj.jsonl"
    )

    result = db.query_policy_trajectory_series(policy="ALNS + Ftsp")
    assert len(result["series"]) == 1
    series = result["series"][0]
    assert series["metric_name"] == "best_cost"
    assert series["x"] == [0, 1, 2, 3]
    assert series["y"] == [50.0, 45.0, 42.0, 40.0]
    assert series["label"].endswith("· d2")


def test_query_trajectory_series_filters_policy_type():
    db.persist_policy_viz_snapshot(
        {"iteration": [0, 1], "best_cost": [10.0, 9.0]},
        "ALNS",
        0,
        1,
        "alns",
        "/tmp/a.jsonl",
    )
    db.persist_policy_viz_snapshot(
        {"generation": [0, 1], "best_cost": [20.0, 18.0]},
        "HGS",
        0,
        1,
        "hgs",
        "/tmp/b.jsonl",
    )

    alns = db.query_policy_trajectory_series(policy_type="alns")
    assert len(alns["series"]) == 1
    assert alns["series"][0]["policy"] == "ALNS"
