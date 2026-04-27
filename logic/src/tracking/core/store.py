"""SQLite storage backend for the WSTracker.

Uses WAL (Write-Ahead Logging) journal mode so multiple processes can read/write
concurrently without corrupting the database. Every public method opens its own
short-lived connection and closes it before returning, providing a simple and
safe pattern for multi-process SQLite access.

Attributes:
    TrackingStore: Persistent SQLite store for experiment tracking data.

Example:
    >>> from logic.src.tracking.core.store import TrackingStore
    >>> store = TrackingStore("tracking.db")
    >>> store.get_or_create_experiment("VRPP-Optimization")
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple


def _safe_json_dumps(value: Any) -> str:
    """JSON-serializes value with a fallback for non-serializable types.

    Args:
        value: The object to serialize.

    Returns:
        str: JSON-formatted string representing the value.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TrackingStore:
    """Persistent SQLite store for experiment tracking data.

    Thread-safe and multi-process safe via SQLite WAL mode. Each operation
    acquires and releases its own connection to avoid long-lived locks and
    ensure robustness in distributed or multi-threaded environments.

    Attributes:
        db_path: Absolute path to the SQLite database file.
        queries: Dictionary holding loaded SQL query templates.
    """

    def __init__(self, db_path: str) -> None:
        """Initializes the storage backend and applies the schema.

        Args:
            db_path: Absolute path to the .db file. If the file does not exist,
                it will be created along with its parent directories.
        """
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
        """Yields a short-lived SQLite connection with autocommit on success.

        Configures the connection for WAL mode and normal synchronous behavior
        to optimize performance while maintaining safety.

        Yields:
            sqlite3.Connection: A managed SQLite connection instance.

        Raises:
            Exception: Re-raises any exception after rolling back the transaction.
        """
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
        """Initializes the database schema and performs required migrations.

        Reads the schema.sql file and executes it. Also applies conditional
        alterations for backward compatibility.
        """
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self._conn() as conn:
            conn.executescript(schema_sql)
            with suppress(sqlite3.OperationalError):
                conn.execute(self.queries["alter_dataset_events"])

    @staticmethod
    def _now() -> str:
        """Returns the current UTC timestamp in ISO 8601 format.

        Returns:
            str: ISO-formatted timestamp in UTC.
        """
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def get_or_create_experiment(self, name: str, description: str = "", tags: Optional[Dict[str, Any]] = None) -> int:
        """Retrieves an existing experiment ID or creates a new record if absent.

        Args:
            name: Human-readable name of the experiment.
            description: Optional text describing the experiment goal/setup.
            tags: Optional dictionary of metadata tags for the experiment.

        Returns:
            int: The unique internal ID of the experiment.
        """
        tags_json = json.dumps(tags or {})
        with self._conn() as conn:
            conn.execute(self.queries["insert_experiment"], (name, self._now(), description, tags_json))
            row = conn.execute(self.queries["get_experiment_id"], (name,)).fetchone()
            return int(row["id"])

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lists all existing experiments in the store.

        Returns:
            List[Dict[str, Any]]: Experiment records containing 'id', 'name',
                'created_at', and metadata.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["list_experiments"]).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self, run_id: str, experiment_id: int, name: Optional[str], run_type: str, artifact_dir: str
    ) -> None:
        """Creates a new tracking run record.

        Args:
            run_id: Unique UUID string identifying the run.
            experiment_id: Internal ID of the parent experiment.
            name: Optional human-readable label for the run.
            run_type: Category identifier (e.g., 'training', 'simulation').
            artifact_dir: Filesystem path to the run's artifact storage.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_run"], (run_id, experiment_id, name, run_type, self._now(), artifact_dir))

    def finish_run(self, run_id: str, status: str = "completed", error: Optional[str] = None) -> None:
        """Marks a run as completed and records the final status.

        Args:
            run_id: UUID string identifying the run.
            status: Terminal status (e.g., 'completed', 'failed', 'killed').
            error: Optional error message or traceback summarizing the failure.
        """
        with self._conn() as conn:
            conn.execute(self.queries["update_run_status"], (status, self._now(), error, run_id))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the full record for a specific run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Optional[Dict[str, Any]]: The run record dictionary, or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(self.queries["get_run"], (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(
        self, experiment_id: Optional[int] = None, run_type: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves a filtered list of runs.

        Args:
            experiment_id: Optional parent experiment internal ID filter.
            run_type: Optional category label filter.
            status: Optional terminal status filter (e.g., 'completed').

        Returns:
            List[Dict[str, Any]]: Matching run records ordered by start time.
        """
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
        """Records a metadata tag for a run.

        Args:
            run_id: UUID string identifying the run.
            key: Tag key identifier.
            value: Tag string value.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_tag"], (run_id, key, value))

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        """Records multiple metadata tags for a run in a single transaction.

        Args:
            run_id: UUID string identifying the run.
            tags: Mapping of tag keys to string values.
        """
        with self._conn() as conn:
            conn.executemany(self.queries["insert_tag"], [(run_id, k, str(v)) for k, v in tags.items()])

    def get_tags(self, run_id: str) -> Dict[str, str]:
        """Retrieves all tags associated with a run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Dict[str, str]: Mapping of retrieved tag keys to their string values.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_tags"], (run_id,)).fetchall()
            return {r["key"]: r["value"] for r in rows}

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Records a configuration parameter.

        Args:
            run_id: UUID string identifying the run.
            key: Parameter key identifier.
            value: Parameter value. Will be JSON-serialized.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_param"], (run_id, key, _safe_json_dumps(value)))

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Records multiple configuration parameters in a single transaction.

        Args:
            run_id: UUID string identifying the run.
            params: Mapping of parameter keys to values.
        """
        try:
            with self._conn() as conn:
                conn.executemany(
                    self.queries["insert_param"], [(run_id, k, _safe_json_dumps(v)) for k, v in params.items()]
                )
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                self._apply_schema()
                with self._conn() as conn:
                    conn.executemany(
                        self.queries["insert_param"], [(run_id, k, _safe_json_dumps(v)) for k, v in params.items()]
                    )
            else:
                raise

    def get_params(self, run_id: str) -> Dict[str, Any]:
        """Retrieves all configuration parameters associated with a run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Dict[str, Any]: Mapping of parameter keys to deserialized values.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_params"], (run_id,)).fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        """Records a single scalar metric value.

        Args:
            run_id: UUID string identifying the run.
            key: Metric identifying key.
            value: Scalar metric value.
            step: Optional logical step index. Defaults to 0.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_metric"], (run_id, key, float(value), step, self._now()))

    def log_metrics_batch(self, run_id: str, metrics: List[Tuple[str, float, int]]) -> None:
        """Records a batch of metric values for improved performance.

        Args:
            run_id: UUID string identifying the run.
            metrics: List of (key, value, step) triples to be inserted.
        """
        now = self._now()
        try:
            with self._conn() as conn:
                conn.executemany(self.queries["insert_metric"], [(run_id, k, float(v), s, now) for k, v, s in metrics])
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                self._apply_schema()
                with self._conn() as conn:
                    conn.executemany(
                        self.queries["insert_metric"], [(run_id, k, float(v), s, now) for k, v, s in metrics]
                    )
            else:
                raise

    def get_metric_history(self, run_id: str, key: str) -> List[Dict[str, Any]]:
        """Retrieves the full evolution history for a specific metric.

        Args:
            run_id: UUID string identifying the run.
            key: Metric identifying key.

        Returns:
            List[Dict[str, Any]]: Historical records ordered by step/time.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_metric_history"], (run_id, key)).fetchall()
            return [dict(r) for r in rows]

    def get_latest_metrics(self, run_id: str) -> Dict[str, float]:
        """Retrieves the most recent value for every metric logged in a run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Dict[str, float]: Mapping of metric keys to their latest logged scalars.
        """
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
        """Registers a local file as an artifact record.

        Args:
            run_id: UUID string identifying the run.
            name: Human-readable label for the artifact.
            path: Relative path to the file within the artifact storage.
            artifact_type: Category identifier (e.g., 'model', 'plot').
            file_hash: Optional SHA-256 hash of the file content.
            size_bytes: Optional file size in bytes.
            metadata: Optional dictionary of additional contextual metadata.
        """
        with self._conn() as conn:
            existing = conn.execute(self.queries["check_artifact"], (run_id, path)).fetchone()
            if existing:
                return
            conn.execute(
                self.queries["insert_artifact"],
                (run_id, name, path, artifact_type, file_hash, size_bytes, self._now(), json.dumps(metadata or {})),
            )

    def get_artifacts(self, run_id: str, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves the metadata for artifacts registered in a run.

        Args:
            run_id: UUID string identifying the run.
            artifact_type: Optional category filter.

        Returns:
            List[Dict[str, Any]]: Matching artifact metadata records.
        """
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
        """Records a data-lifecycle event for lineage tracking.

        Args:
            run_id: UUID string identifying the run.
            event_type: Category (e.g., 'load', 'save', 'mutate').
            file_path: Path to the associated file.
            file_hash: Current SHA-256 hash of the file.
            prev_hash: Previous known SHA-256 hash for version tracking.
            size_bytes: Current file size in bytes.
            shape: Optional logical shape of the dataset if applicable.
            metadata: Optional dictionary of extra experiment-specific info.
        """
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
        """Retrieves the data-lifecycle events for a specific run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            List[Dict[str, Any]]: Event records in chronological order.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_dataset_events"], (run_id,)).fetchall()
            return [dict(r) for r in rows]
