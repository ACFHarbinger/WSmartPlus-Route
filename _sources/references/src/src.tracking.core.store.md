# {py:mod}`src.tracking.core.store`

```{py:module} src.tracking.core.store
```

```{autodoc2-docstring} src.tracking.core.store
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrackingStore <src.tracking.core.store.TrackingStore>`
  - ```{autodoc2-docstring} src.tracking.core.store.TrackingStore
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_safe_json_dumps <src.tracking.core.store._safe_json_dumps>`
  - ```{autodoc2-docstring} src.tracking.core.store._safe_json_dumps
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SCHEMA_SQL <src.tracking.core.store._SCHEMA_SQL>`
  - ```{autodoc2-docstring} src.tracking.core.store._SCHEMA_SQL
    :summary:
    ```
````

### API

````{py:function} _safe_json_dumps(value: typing.Any) -> str
:canonical: src.tracking.core.store._safe_json_dumps

```{autodoc2-docstring} src.tracking.core.store._safe_json_dumps
```
````

````{py:data} _SCHEMA_SQL
:canonical: src.tracking.core.store._SCHEMA_SQL
:value: <Multiline-String>

```{autodoc2-docstring} src.tracking.core.store._SCHEMA_SQL
```

````

`````{py:class} TrackingStore(db_path: str)
:canonical: src.tracking.core.store.TrackingStore

```{autodoc2-docstring} src.tracking.core.store.TrackingStore
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.__init__
```

````{py:method} _conn() -> typing.Generator[sqlite3.Connection, None, None]
:canonical: src.tracking.core.store.TrackingStore._conn

```{autodoc2-docstring} src.tracking.core.store.TrackingStore._conn
```

````

````{py:method} _apply_schema() -> None
:canonical: src.tracking.core.store.TrackingStore._apply_schema

```{autodoc2-docstring} src.tracking.core.store.TrackingStore._apply_schema
```

````

````{py:method} _now() -> str
:canonical: src.tracking.core.store.TrackingStore._now
:staticmethod:

```{autodoc2-docstring} src.tracking.core.store.TrackingStore._now
```

````

````{py:method} get_or_create_experiment(name: str, description: str = '', tags: typing.Optional[typing.Dict[str, typing.Any]] = None) -> int
:canonical: src.tracking.core.store.TrackingStore.get_or_create_experiment

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_or_create_experiment
```

````

````{py:method} list_experiments() -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.store.TrackingStore.list_experiments

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.list_experiments
```

````

````{py:method} create_run(run_id: str, experiment_id: int, name: typing.Optional[str], run_type: str, artifact_dir: str) -> None
:canonical: src.tracking.core.store.TrackingStore.create_run

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.create_run
```

````

````{py:method} finish_run(run_id: str, status: str = 'completed', error: typing.Optional[str] = None) -> None
:canonical: src.tracking.core.store.TrackingStore.finish_run

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.finish_run
```

````

````{py:method} get_run(run_id: str) -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.store.TrackingStore.get_run

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_run
```

````

````{py:method} list_runs(experiment_id: typing.Optional[int] = None, run_type: typing.Optional[str] = None, status: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.store.TrackingStore.list_runs

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.list_runs
```

````

````{py:method} set_tag(run_id: str, key: str, value: str) -> None
:canonical: src.tracking.core.store.TrackingStore.set_tag

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.set_tag
```

````

````{py:method} set_tags(run_id: str, tags: typing.Dict[str, str]) -> None
:canonical: src.tracking.core.store.TrackingStore.set_tags

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.set_tags
```

````

````{py:method} get_tags(run_id: str) -> typing.Dict[str, str]
:canonical: src.tracking.core.store.TrackingStore.get_tags

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_tags
```

````

````{py:method} log_param(run_id: str, key: str, value: typing.Any) -> None
:canonical: src.tracking.core.store.TrackingStore.log_param

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.log_param
```

````

````{py:method} log_params(run_id: str, params: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.core.store.TrackingStore.log_params

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.log_params
```

````

````{py:method} get_params(run_id: str) -> typing.Dict[str, typing.Any]
:canonical: src.tracking.core.store.TrackingStore.get_params

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_params
```

````

````{py:method} log_metric(run_id: str, key: str, value: float, step: int = 0) -> None
:canonical: src.tracking.core.store.TrackingStore.log_metric

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.log_metric
```

````

````{py:method} log_metrics_batch(run_id: str, metrics: typing.List[typing.Tuple[str, float, int]]) -> None
:canonical: src.tracking.core.store.TrackingStore.log_metrics_batch

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.log_metrics_batch
```

````

````{py:method} get_metric_history(run_id: str, key: str) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.store.TrackingStore.get_metric_history

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_metric_history
```

````

````{py:method} get_latest_metrics(run_id: str) -> typing.Dict[str, float]
:canonical: src.tracking.core.store.TrackingStore.get_latest_metrics

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_latest_metrics
```

````

````{py:method} log_artifact(run_id: str, name: str, path: str, artifact_type: str = 'file', file_hash: typing.Optional[str] = None, size_bytes: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.core.store.TrackingStore.log_artifact

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.log_artifact
```

````

````{py:method} get_artifacts(run_id: str, artifact_type: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.store.TrackingStore.get_artifacts

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_artifacts
```

````

````{py:method} log_dataset_event(run_id: str, event_type: str, file_path: typing.Optional[str] = None, file_hash: typing.Optional[str] = None, prev_hash: typing.Optional[str] = None, size_bytes: typing.Optional[int] = None, num_samples: typing.Optional[int] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None
:canonical: src.tracking.core.store.TrackingStore.log_dataset_event

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.log_dataset_event
```

````

````{py:method} get_dataset_events(run_id: str) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.store.TrackingStore.get_dataset_events

```{autodoc2-docstring} src.tracking.core.store.TrackingStore.get_dataset_events
```

````

`````
