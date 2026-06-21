"""Unit tests for the tracking database stats and metrics subcommands."""

import os
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timedelta, timezone

import logic.src.tracking.database.shared as shared
import logic.src.tracking.database.cmd_stats as cmd_stats

SCHEMA_PATH = Path(__file__).parents[3] / "src/tracking/core/schema.sql"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Fixture to initialize a temporary tracking database with dummy records for stats."""
    db_file = tmp_path / "test_tracking.db"
    db_path_str = str(db_file.resolve())

    # Patch DB_PATH in both shared and cmd_stats
    monkeypatch.setattr(shared, "DB_PATH", db_path_str)
    monkeypatch.setattr(cmd_stats, "DB_PATH", db_path_str)

    # Initialize the database using the schema.sql
    conn = sqlite3.connect(db_path_str)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)

    # Populate dummy data
    # 1. Experiment
    conn.execute(
        "INSERT INTO experiments (id, name, created_at, description) "
        "VALUES (1, 'test_exp', '2026-06-21T12:00:00', 'Test description')"
    )

    # 2. Runs
    now = datetime.now(timezone.utc)
    start_time = (now - timedelta(minutes=30)).isoformat()
    end_time = now.isoformat()

    conn.execute(
        "INSERT INTO runs (id, experiment_id, name, status, run_type, start_time, end_time) "
        "VALUES (?, 1, 'run_1', 'completed', 'generic', ?, ?)",
        ("run_1_uuid_32_chars_long_exactly_abc", start_time, end_time)
    )

    # 3. Metrics
    conn.execute(
        "INSERT INTO metrics (run_id, key, value, step, timestamp) "
        "VALUES (?, 'profit', 100.0, 0, ?)",
        ("run_1_uuid_32_chars_long_exactly_abc", end_time)
    )
    conn.execute(
        "INSERT INTO metrics (run_id, key, value, step, timestamp) "
        "VALUES (?, 'profit', 120.0, 1, ?)",
        ("run_1_uuid_32_chars_long_exactly_abc", end_time)
    )

    # 4. Artifacts
    conn.execute(
        "INSERT INTO artifacts (run_id, name, path, artifact_type, size_bytes, created_at) "
        "VALUES (?, 'art', 'path/to/art', 'file', 1024, ?)",
        ("run_1_uuid_32_chars_long_exactly_abc", end_time)
    )

    # 5. Dataset Events
    conn.execute(
        "INSERT INTO dataset_events (run_id, event_type, timestamp) VALUES (?, 'load', ?)",
        ("run_1_uuid_32_chars_long_exactly_abc", end_time)
    )

    conn.commit()
    conn.close()

    return db_path_str


@pytest.mark.unit
def test_human_bytes():
    assert cmd_stats._human_bytes(512) == "512.0 B"
    assert cmd_stats._human_bytes(1024) == "1.0 KB"
    assert cmd_stats._human_bytes(1024 * 1024) == "1.0 MB"
    assert cmd_stats._human_bytes(1024 * 1024 * 1024) == "1.0 GB"
    assert cmd_stats._human_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"


@pytest.mark.unit
def test_human_duration():
    assert cmd_stats._human_duration(45.0) == "45.0s"
    assert cmd_stats._human_duration(120.0) == "2.0m"
    assert cmd_stats._human_duration(7200.0) == "2.0h"


@pytest.mark.unit
def test_sparkbar():
    assert cmd_stats._sparkbar(0, 10, width=5) == "░░░░░"
    assert cmd_stats._sparkbar(5, 10, width=5) == "██░░░"
    assert cmd_stats._sparkbar(10, 10, width=5) == "█████"
    assert cmd_stats._sparkbar(5, 0, width=5) == "░░░░░"


@pytest.mark.unit
def test_stats_database(temp_db, capsys):
    cmd_stats.stats_database()
    captured = capsys.readouterr()
    assert "WSmart-Route Tracking Database — Statistics" in captured.out
    assert "experiments" in captured.out
    assert "test_exp" in captured.out
    assert "Run Duration" in captured.out
    assert "Top Metrics" in captured.out
    assert "Artifact Types" in captured.out
    assert "Dataset Events" in captured.out
    assert "Run Activity" in captured.out


@pytest.mark.unit
def test_stats_database_with_experiment(temp_db, capsys):
    cmd_stats.stats_database(experiment_name="test_exp")
    captured = capsys.readouterr()
    assert "[test_exp]" in captured.out


@pytest.mark.unit
def test_stats_database_not_found(monkeypatch, capsys):
    monkeypatch.setattr(shared, "DB_PATH", "non_existent_db_file.db")
    monkeypatch.setattr(cmd_stats, "DB_PATH", "non_existent_db_file.db")
    cmd_stats.stats_database()
    captured = capsys.readouterr()
    assert "Tracking database not found." in captured.out


@pytest.mark.unit
def test_metrics_summary_all(temp_db, capsys):
    cmd_stats.metrics_summary()
    captured = capsys.readouterr()
    assert "Metric Summary" in captured.out
    assert "profit" in captured.out


@pytest.mark.unit
def test_metrics_summary_all_no_metrics(temp_db, capsys):
    # Clear metrics table
    conn = sqlite3.connect(temp_db)
    conn.execute("DELETE FROM metrics")
    conn.commit()
    conn.close()

    cmd_stats.metrics_summary()
    captured = capsys.readouterr()
    assert "No metrics recorded yet." in captured.out


@pytest.mark.unit
def test_metrics_summary_key(temp_db, capsys):
    cmd_stats.metrics_summary(key="profit")
    captured = capsys.readouterr()
    assert "Metric Detail: profit" in captured.out
    assert "run_1" in captured.out or "run_1_uuid" in captured.out


@pytest.mark.unit
def test_metrics_summary_key_not_found(temp_db, capsys):
    cmd_stats.metrics_summary(key="non_existent")
    captured = capsys.readouterr()
    assert "No data found for metric 'non_existent'." in captured.out


@pytest.mark.unit
def test_metrics_summary_not_found(monkeypatch, capsys):
    monkeypatch.setattr(shared, "DB_PATH", "non_existent_db_file.db")
    monkeypatch.setattr(cmd_stats, "DB_PATH", "non_existent_db_file.db")
    cmd_stats.metrics_summary()
    captured = capsys.readouterr()
    assert "Tracking database not found." in captured.out
