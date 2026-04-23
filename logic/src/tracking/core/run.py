"""Run lifecycle management for the WSTracker.

Provides the core recording interface for tracking individual experiments.
Manages buffered metric writes, parameter logging, tag management,
artifact registration, and dataset lifecycle event tracking.

Attributes:
    get_active_run: Function to retrieve the current process-level active run.
    set_active_run: Function to set the process-level active run.
    Run: Main class for interacting with an individual tracked run.

Example:
    >>> from logic.src.tracking.core.run import Run
    >>> from logic.src.tracking.core.store import TrackingStore
    >>> store = TrackingStore("tracking.db")
    >>> with Run("uuid-123", store, "./artifacts") as run:
    ...     run.log_param("lr", 0.001)
    ...     run.log_metric("loss", 0.5, step=1)
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import Literal

from logic.src.tracking.core.store import TrackingStore
from logic.src.tracking.validation.hashing import hash_file

# ---------------------------------------------------------------------------
# Module-level active-run registry (one per process)
# ---------------------------------------------------------------------------

_active_run: Optional["Run"] = None
_registry_lock = threading.Lock()


def get_active_run() -> Optional["Run"]:
    """Returns the active Run instance for the current process.

    Returns:
        Optional[Run]: The current process-global active run, or None if no
            run is active.
    """
    return _active_run


def set_active_run(run: Optional["Run"]) -> None:
    """Sets the active Run instance for the current process.

    This is usually called automatically by the Run context manager.

    Args:
        run: The Run instance to register as active, or None to clear the registry.
    """
    global _active_run
    with _registry_lock:
        _active_run = run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary using separator-joined keys.

    Args:
        d: The nested dictionary to flatten.
        prefix: Current key prefix for recursive calls.
        sep: Separator character between nested levels.

    Returns:
        Dict[str, Any]: A single-level dictionary with flattened keys.
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key, sep))
        else:
            result[key] = v
    return result


def _safe_float(v: Any) -> Optional[float]:
    """Safely converts a value to a float, returning None on failure or overflow.

    Args:
        v: The value to convert.

    Returns:
        Optional[float]: The converted float value, or None if invalid.
    """
    try:
        fval = float(v)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return fval
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run:
    """A single tracked experiment run.

    Provides high-level methods to log metrics, parameters, tags, artifacts, and
    dataset lifecycle events. Metric writes are buffered in memory and flushed
    to SQLite in batches to maximize I/O performance.

    This class is thread-safe; multiple threads may log concurrently.

    Attributes:
        run_id: Unique UUID string identifying the run in the database.
        artifact_dir: Root directory where this run's artifacts are stored.
    """

    def __init__(
        self,
        run_id: str,
        store: TrackingStore,
        artifact_dir: str,
        buffer_size: int = 200,
    ) -> None:
        """Initializes a new tracking run.

        Args:
            run_id: UUID string identifying the run.
            store: Persistent storage backend instance.
            artifact_dir: Local directory for storing artifact files.
            buffer_size: Number of metrics to buffer before an automatic flush.
        """
        self.run_id = run_id
        self._store = store
        self.artifact_dir = artifact_dir
        self._buffer_size = buffer_size

        self._metric_buffer: List[Tuple[str, float, int]] = []
        self._buffer_lock = threading.Lock()

        # path -> last observed SHA-256 for change detection
        self._file_hashes: Dict[str, str] = {}
        self._closed = False

        # Optional secondary logging sinks (e.g. MLflowBridge).
        # Each sink must expose log_metric(key, value, step),
        # log_params(params), log_artifact(path), and finish(status).
        self._sinks: List[Any] = []

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def add_sink(self, sink: Any) -> "Run":
        """Attaches a secondary logging sink to this run.

        The sink receives forwarded calls for every metric, param, and
        artifact logged. Sinks are useful for mirroring data to external
        services like MLflow or W&B.

        Args:
            sink: Object implementing log_metric, log_params, log_artifact,
                and finish methods.

        Returns:
            Run: Self for method chaining.
        """
        self._sinks.append(sink)
        return self

    def set_tag(self, key: str, value: Any) -> "Run":
        """Attaches a metadata tag to this run.

        Args:
            key: Semantic name for the tag.
            value: Tag value (will be cast to string).

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            self._store.set_tag(self.run_id, key, str(value))
        return self

    def set_tags(self, tags: Dict[str, Any]) -> "Run":
        """Attaches multiple metadata tags to this run.

        Args:
            tags: Dictionary of tag keys and values.

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            self._store.set_tags(self.run_id, {k: str(v) for k, v in tags.items()})
        return self

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> "Run":
        """Records a single configuration parameter.

        Args:
            key: Name of the parameter.
            value: Parameter value.

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            self._store.log_param(self.run_id, key, value)
        return self

    def log_params(self, params: Dict[str, Any]) -> "Run":
        """Records multiple configuration parameters, flattening nested dicts.

        Args:
            params: Dictionary of parameters to log.

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            flat = _flatten_dict(params)
            self._store.log_params(self.run_id, flat)
            for sink in self._sinks:
                with contextlib.suppress(Exception):
                    sink.log_params(flat)
        return self

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: Any, step: int = 0) -> "Run":
        """Logs a single scalar metric value.

        Args:
            key: Name of the metric.
            value: Scalar value to log (must be convertible to float).
            step: Training or simulation step index.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed:
            return self
        fval = _safe_float(value)
        if fval is None:
            return self
        with self._buffer_lock:
            self._metric_buffer.append((key, fval, step))
            if len(self._metric_buffer) >= self._buffer_size:
                self._flush_metrics_locked()
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.log_metric(key, fval, step)
        return self

    def log_metrics(self, metrics: Dict[str, Any], step: int = 0) -> "Run":
        """Logs a dictionary of scalar metric values.

        Args:
            metrics: Mapping of metric keys to values.
            step: Training or simulation step index.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed:
            return self
        with self._buffer_lock:
            for k, v in metrics.items():
                fval = _safe_float(v)
                if fval is not None:
                    self._metric_buffer.append((k, fval, step))
            if len(self._metric_buffer) >= self._buffer_size:
                self._flush_metrics_locked()
        return self

    def flush(self) -> "Run":
        """Force-flushes any buffered metrics to the storage backend.

        Returns:
            Run: Self for method chaining.
        """
        with self._buffer_lock:
            self._flush_metrics_locked()
        return self

    def _flush_metrics_locked(self) -> None:
        """Internal helper to flush the metric buffer.

        Caller must hold `_buffer_lock`.
        """
        if self._metric_buffer:
            self._store.log_metrics_batch(self.run_id, self._metric_buffer)
            self._metric_buffer.clear()

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Run":
        """Registers a local file as an artifact of this run.

        Args:
            path: Local path to the artifact file.
            name: Human-readable name (defaults to file basename).
            artifact_type: Semantic category (e.g., 'model', 'plot').
            metadata: Additional JSON-serializable metadata.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed or not os.path.exists(path):
            return self
        self._store.log_artifact(
            self.run_id,
            name or os.path.basename(path),
            path,
            artifact_type=artifact_type,
            file_hash=hash_file(path),
            size_bytes=os.path.getsize(path),
            metadata=metadata,
        )
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.log_artifact(path)
        return self

    def log_artifacts_dir(
        self,
        dir_path: str,
        artifact_type: str = "file",
    ) -> "Run":
        """Registers every file in a directory as an artifact.

        Note: This is non-recursive.

        Args:
            dir_path: Path to the directory.
            artifact_type: Semantic category applied to all files.

        Returns:
            Run: Self for method chaining.
        """
        if not os.path.isdir(dir_path):
            return self
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath):
                self.log_artifact(fpath, artifact_type=artifact_type)
        return self

    # ------------------------------------------------------------------
    # Dataset events
    # ------------------------------------------------------------------

    def log_dataset_event(
        self,
        event_type: str,
        file_path: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Run":
        """Records a data-lifecycle event (e.g., loading, mutation).

        Args:
            event_type: Semantic event (load, save, generate, hash_change).
            file_path: Path to the associated data file if applicable.
            shape: Logical shape of the dataset.
            metadata: Extra contextual information.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed:
            return self

        file_hash: Optional[str] = None
        prev_hash: Optional[str] = None
        size_bytes: Optional[int] = None

        if file_path and os.path.exists(file_path):
            file_hash = hash_file(file_path)
            size_bytes = os.path.getsize(file_path)
            prev_hash = self._file_hashes.get(file_path)
            self._file_hashes[file_path] = file_hash  # type: ignore[assignment]

        self._store.log_dataset_event(
            self.run_id,
            event_type=event_type,
            file_path=file_path,
            file_hash=file_hash,
            prev_hash=prev_hash,
            size_bytes=size_bytes,
            shape=shape,
            metadata=metadata,
        )
        return self

    def watch_file(self, file_path: str) -> "Run":
        """Registers the current SHA-256 of a file for change detection.

        Args:
            file_path: Path to the file to monitor.

        Returns:
            Run: Self for method chaining.
        """
        if os.path.exists(file_path):
            h = hash_file(file_path)
            if h is not None:
                self._file_hashes[file_path] = h
        return self

    def check_file_changed(self, file_path: str) -> bool:
        """Determines if a watched file has changed on disk.

        Updating the baseline hash on every call ensures subsequent checks
        measure drift from the most recent confirmed state.

        Args:
            file_path: Path to the file to check.

        Returns:
            bool: True if the file hash has changed since the last observation.
        """
        if not os.path.exists(file_path):
            return False
        current = hash_file(file_path)
        prev = self._file_hashes.get(file_path)
        if current is not None:
            self._file_hashes[file_path] = current
        changed = prev is None or current != prev
        if changed and prev is not None:
            # Observed change — log it automatically
            self._store.log_dataset_event(
                self.run_id,
                event_type="hash_change",
                file_path=file_path,
                file_hash=current,
                prev_hash=prev,
                size_bytes=os.path.getsize(file_path),
            )
        return changed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self, status: str = "completed", error: Optional[str] = None) -> None:
        """Concludes the run, flushing buffers and updating state.

        Args:
            status: Final status (completed, failed, killed).
            error: Optional error message summarizing the failure.
        """
        if not self._closed:
            self.flush()
            self._store.finish_run(self.run_id, status=status, error=error)
            self._closed = True
            for sink in self._sinks:
                with contextlib.suppress(Exception):
                    sink.finish(status)
            if get_active_run() is self:
                set_active_run(None)

    # Context-manager support
    def __enter__(self) -> "Run":
        """Enters the tracking context, registering this as the active run.

        Returns:
            Run: The active run instance.
        """
        set_active_run(self)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        _exc_tb: Any,
    ) -> Literal[False]:
        """Exits the context, automatically finishing the run.

        Handles exception logging if the context was exited via an error.

        Args:
            exc_type: Type of the exception if one occurred.
            exc_val: Exception instance.
            _exc_tb: Traceback object.

        Returns:
            Literal[False]: Always returns False to propagate exceptions.
        """
        if exc_type is not None:
            self.finish(status="failed", error=str(exc_val))
        else:
            self.finish(status="completed")
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Retrieves all configuration parameters logged for this run.

        Returns:
            Dict[str, Any]: Dictionary mapping parameter keys to values.
        """
        return self._store.get_params(self.run_id)

    def get_tags(self) -> Dict[str, str]:
        """Retrieves all metadata tags logged for this run.

        Returns:
            Dict[str, str]: Dictionary mapping tag keys to values.
        """
        return self._store.get_tags(self.run_id)

    def get_latest_metrics(self) -> Dict[str, float]:
        """Retrieves the most recent value for every logged metric.

        Returns:
            Dict[str, float]: Dictionary mapping metric keys to their latest scalars.
        """
        return self._store.get_latest_metrics(self.run_id)

    def get_metric_history(self, key: str) -> List[Dict[str, Any]]:
        """Retrieves the full evolution history for a specific metric.

        Args:
            key: Semantic name of the metric to query.

        Returns:
            List[Dict[str, Any]]: Step-indexed records containing 'value',
                'step', and 'timestamp'.
        """
        return self._store.get_metric_history(self.run_id, key)

    def get_artifacts(self, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves Registered artifacts for this run.

        Args:
            artifact_type: Optional filter by semantic category.

        Returns:
            List[Dict[str, Any]]: Artifact records mapping attributes like
                'name', 'path', and 'metadata'.
        """
        return self._store.get_artifacts(self.run_id, artifact_type=artifact_type)

    def get_dataset_events(self) -> List[Dict[str, Any]]:
        """Retrieves all data-lifecycle events in chronological order.

        Returns:
            List[Dict[str, Any]]: Event records containing type, path, and context.
        """
        return self._store.get_dataset_events(self.run_id)

    def __repr__(self) -> str:
        """Returns a string representation of the Run state.

        Returns:
            str: Developer-readable string summary.
        """
        return f"Run(id={self.run_id!r}, closed={self._closed})"
