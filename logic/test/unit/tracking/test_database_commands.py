"""Unit tests for the tracking database management commands."""

import os
import sys
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import logic.src.tracking.database.shared as shared
import logic.src.tracking.database.commands as commands

# Determine schema path relative to this test file
SCHEMA_PATH = Path(__file__).parents[3] / "src/tracking/core/schema.sql"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Fixture to initialize a temporary tracking database with dummy records."""
    db_file = tmp_path / "test_tracking.db"
    db_path_str = str(db_file.resolve())

    # Patch DB_PATH in both shared and commands
    monkeypatch.setattr(shared, "DB_PATH", db_path_str)
    monkeypatch.setattr(commands, "DB_PATH", db_path_str)

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
    # run_1: failed, older than 30 days
    conn.execute(
        "INSERT INTO runs (id, experiment_id, name, status, run_type, start_time) "
        "VALUES (?, 1, 'run_1', 'failed', 'generic', '2026-05-01T12:00:00')",
        ("run_1_uuid_32_chars_long_exactly_abc",)
    )
    # run_2: completed, recent
    conn.execute(
        "INSERT INTO runs (id, experiment_id, name, status, run_type, start_time) "
        "VALUES (?, 1, 'run_2', 'completed', 'generic', '2026-06-20T12:00:00')",
        ("run_2_uuid_32_chars_long_exactly_xyz",)
    )

    # 3. Tags
    conn.execute(
        "INSERT INTO run_tags (run_id, key, value) VALUES (?, 'tag_k', 'tag_v')",
        ("run_1_uuid_32_chars_long_exactly_abc",)
    )

    # 4. Params
    conn.execute(
        "INSERT INTO params (run_id, key, value) VALUES (?, 'param_k', '\"param_v\"')",
        ("run_1_uuid_32_chars_long_exactly_abc",)
    )

    # 5. Metrics
    conn.execute(
        "INSERT INTO metrics (run_id, key, value, step, timestamp) "
        "VALUES (?, 'profit', 100.0, 0, '2026-06-21T12:00:00')",
        ("run_1_uuid_32_chars_long_exactly_abc",)
    )

    # 6. Artifacts
    conn.execute(
        "INSERT INTO artifacts (run_id, name, path, artifact_type, created_at) "
        "VALUES (?, 'art', 'path/to/art', 'file', '2026-06-21T12:00:00')",
        ("run_1_uuid_32_chars_long_exactly_abc",)
    )

    # 7. Dataset Events
    conn.execute(
        "INSERT INTO dataset_events (run_id, event_type, timestamp) VALUES (?, 'load', '2026-06-21T12:00:00')",
        ("run_1_uuid_32_chars_long_exactly_abc",)
    )

    conn.commit()
    conn.close()

    return db_path_str


@pytest.mark.unit
def test_inspect_database(temp_db, capsys):
    commands.inspect_database()
    captured = capsys.readouterr()
    assert "WSmart-Route Tracking Database" in captured.out
    assert "test_exp" in captured.out
    assert "run_1" in captured.out or "run_1_uuid" in captured.out


@pytest.mark.unit
def test_compact_database(temp_db, capsys):
    commands.compact_database()
    captured = capsys.readouterr()
    assert "Integrity   : ✅ ok" in captured.out
    assert "Database compacted." in captured.out


@pytest.mark.unit
def test_clean_database(temp_db, capsys):
    commands.clean_database()
    captured = capsys.readouterr()
    assert "Database cleaned" in captured.out or "cleaned" in captured.out


@pytest.mark.unit
def test_prune_database_dry_run(temp_db, capsys):
    commands.prune_database(older_than_days=10, status="failed", dry_run=True)
    captured = capsys.readouterr()
    assert "Would remove 1 run(s)" in captured.out
    assert "run_1" in captured.out
    assert "(dry-run — no changes made)" in captured.out


@pytest.mark.unit
def test_prune_database_execute(temp_db, capsys):
    commands.prune_database(older_than_days=10, status="failed", dry_run=False)
    captured = capsys.readouterr()
    assert "Removing 1 run(s)" in captured.out
    assert "Pruned 1 run(s)" in captured.out


@pytest.mark.unit
def test_prune_database_no_match(temp_db, capsys):
    commands.prune_database(older_than_days=10, status="non_existent", dry_run=False)
    captured = capsys.readouterr()
    assert "No runs match the prune criteria" in captured.out


@pytest.mark.unit
def test_export_run_stdout(temp_db, capsys):
    commands.export_run(run_id="run_1_uuid_32_chars_long_exactly_abc")
    captured = capsys.readouterr()
    assert "run_1" in captured.out
    assert "param_v" in captured.out


@pytest.mark.unit
def test_export_run_prefix(temp_db, capsys):
    commands.export_run(run_id="run_1")
    captured = capsys.readouterr()
    assert "run_1" in captured.out
    assert "param_v" in captured.out


@pytest.mark.unit
def test_export_run_latest(temp_db, capsys):
    commands.export_run(experiment_name="test_exp", latest=True)
    captured = capsys.readouterr()
    # Should resolve to the latest run in experiment (run_2)
    assert "run_2" in captured.out


@pytest.mark.unit
def test_export_run_to_file(temp_db, tmp_path, capsys):
    out_file = tmp_path / "export.json"
    commands.export_run(run_id="run_1_uuid_32_chars_long_exactly_abc", output=str(out_file.resolve()))
    assert out_file.exists()
    import json
    with open(out_file) as f:
        data = json.load(f)
    assert data["name"] == "run_1"


@pytest.mark.unit
def test_export_run_not_found(temp_db):
    with pytest.raises(SystemExit):
        commands.export_run(run_id="non_existent")


@pytest.mark.unit
def test_export_run_latest_no_runs(temp_db):
    # Delete all runs to trigger the exit
    conn = sqlite3.connect(temp_db)
    conn.execute("DELETE FROM runs")
    conn.commit()
    conn.close()

    with pytest.raises(SystemExit):
        commands.export_run(experiment_name="test_exp", latest=True)


@pytest.mark.unit
def test_export_run_missing_args(temp_db):
    with pytest.raises(SystemExit):
        commands.export_run(run_id="")


@pytest.mark.unit
def test_database_not_found_handling(monkeypatch, capsys):
    # Set to a non-existent database file
    monkeypatch.setattr(shared, "DB_PATH", "non_existent_db_file.db")
    monkeypatch.setattr(commands, "DB_PATH", "non_existent_db_file.db")

    commands.inspect_database()
    captured = capsys.readouterr()
    assert "Tracking database not found." in captured.out

    commands.clean_database()
    captured = capsys.readouterr()
    assert "Tracking database not found." in captured.out

    commands.compact_database()
    captured = capsys.readouterr()
    assert "Tracking database not found." in captured.out

    commands.prune_database()
    captured = capsys.readouterr()
    assert "Tracking database not found." in captured.out

    with pytest.raises(SystemExit):
        commands.export_run(run_id="run_1")
