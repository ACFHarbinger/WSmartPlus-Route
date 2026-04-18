"""SQLite storage backend for the WSTracker.

Uses WAL journal mode so multiple processes can read/write concurrently
without corrupting the database.  Every public method opens its own
short-lived connection and closes it before returning, which is the
simplest safe pattern for multi-process SQLite access.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple


def _safe_json_dumps(value: Any) -> str:
    """JSON-serialise *value*, falling back to ``str()`` for non-serialisable types."""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TrackingStore:
    """Persistent SQLite store for experiment tracking data.

    Thread-safe and multi-process safe via SQLite WAL mode.
    Each operation acquires and releases its own connection.

    Args:
        db_path: Absolute path to the ``.db`` file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Load queries dynamically to decouple SQL from Python
        queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.json")
        with open(queries_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        self._apply_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a short-lived SQLite connection with autocommit on success."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous  = NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _apply_schema(self) -> None:
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self._conn() as conn:
            conn.executescript(schema_sql)
            with suppress(sqlite3.OperationalError):
                conn.execute(self.queries["alter_dataset_events"])

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def get_or_create_experiment(self, name: str, description: str = "", tags: Optional[Dict[str, Any]] = None) -> int:
        tags_json = json.dumps(tags or {})
        with self._conn() as conn:
            conn.execute(self.queries["insert_experiment"], (name, self._now(), description, tags_json))
            row = conn.execute(self.queries["get_experiment_id"], (name,)).fetchone()
            return int(row["id"])

    def list_experiments(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(self.queries["list_experiments"]).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self, run_id: str, experiment_id: int, name: Optional[str], run_type: str, artifact_dir: str
    ) -> None:
        with self._conn() as conn:
            conn.execute(self.queries["insert_run"], (run_id, experiment_id, name, run_type, self._now(), artifact_dir))

    def finish_run(self, run_id: str, status: str = "completed", error: Optional[str] = None) -> None:
        with self._conn() as conn:
            conn.execute(self.queries["update_run_status"], (status, self._now(), error, run_id))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(self.queries["get_run"], (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(
        self, experiment_id: Optional[int] = None, run_type: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = self.queries["list_runs_base"]
        args: List[Any] = []
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            args.append(experiment_id)
        if run_type is not None:
            query += " AND run_type = ?"
            args.append(run_type)
        if status is not None:
            query += " AND status = ?"
            args.append(status)
        query += " ORDER BY start_time DESC"

        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Tags & Params
    # ------------------------------------------------------------------

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(self.queries["insert_tag"], (run_id, key, value))

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        with self._conn() as conn:
            conn.executemany(self.queries["insert_tag"], [(run_id, k, str(v)) for k, v in tags.items()])

    def get_tags(self, run_id: str) -> Dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_tags"], (run_id,)).fetchall()
            return {r["key"]: r["value"] for r in rows}

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        with self._conn() as conn:
            conn.execute(self.queries["insert_param"], (run_id, key, _safe_json_dumps(value)))

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.executemany(
                self.queries["insert_param"], [(run_id, k, _safe_json_dumps(v)) for k, v in params.items()]
            )

    def get_params(self, run_id: str) -> Dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_params"], (run_id,)).fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        with self._conn() as conn:
            conn.execute(self.queries["insert_metric"], (run_id, key, float(value), step, self._now()))

    def log_metrics_batch(self, run_id: str, metrics: List[Tuple[str, float, int]]) -> None:
        now = self._now()
        with self._conn() as conn:
            conn.executemany(self.queries["insert_metric"], [(run_id, k, float(v), s, now) for k, v, s in metrics])

    def get_metric_history(self, run_id: str, key: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_metric_history"], (run_id, key)).fetchall()
            return [dict(r) for r in rows]

    def get_latest_metrics(self, run_id: str) -> Dict[str, float]:
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_latest_metrics"], (run_id,)).fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Artifacts & Dataset Events
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        artifact_type: str = "file",
        file_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._conn() as conn:
            existing = conn.execute(self.queries["check_artifact"], (run_id, path)).fetchone()
            if existing:
                return
            conn.execute(
                self.queries["insert_artifact"],
                (run_id, name, path, artifact_type, file_hash, size_bytes, self._now(), json.dumps(metadata or {})),
            )

    def get_artifacts(self, run_id: str, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query = self.queries["get_artifacts_base"]
        args: List[Any] = [run_id]
        if artifact_type:
            query += " AND artifact_type = ?"
            args.append(artifact_type)
        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
            return [dict(r) for r in rows]

    def log_dataset_event(
        self,
        run_id: str,
        event_type: str,
        file_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        prev_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                self.queries["insert_dataset_event"],
                (
                    run_id,
                    event_type,
                    file_path,
                    file_hash,
                    prev_hash,
                    size_bytes,
                    str(tuple(shape)) if shape is not None else None,
                    json.dumps(metadata or {}),
                    self._now(),
                ),
            )

    def get_dataset_events(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_dataset_events"], (run_id,)).fetchall()
            return [dict(r) for r in rows]
