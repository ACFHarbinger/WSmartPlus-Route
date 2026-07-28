# {py:mod}`src.tracking.logging.modules.policy_telemetry_db`

```{py:module} src.tracking.logging.modules.policy_telemetry_db
```

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_conn <src.tracking.logging.modules.policy_telemetry_db._conn>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._conn
    :summary:
    ```
* - {py:obj}`_ensure_schema <src.tracking.logging.modules.policy_telemetry_db._ensure_schema>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._ensure_schema
    :summary:
    ```
* - {py:obj}`_series_tail <src.tracking.logging.modules.policy_telemetry_db._series_tail>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._series_tail
    :summary:
    ```
* - {py:obj}`extract_final_metric <src.tracking.logging.modules.policy_telemetry_db.extract_final_metric>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.extract_final_metric
    :summary:
    ```
* - {py:obj}`_step_count <src.tracking.logging.modules.policy_telemetry_db._step_count>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._step_count
    :summary:
    ```
* - {py:obj}`_run_label_from_path <src.tracking.logging.modules.policy_telemetry_db._run_label_from_path>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._run_label_from_path
    :summary:
    ```
* - {py:obj}`_upsert_run <src.tracking.logging.modules.policy_telemetry_db._upsert_run>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._upsert_run
    :summary:
    ```
* - {py:obj}`persist_policy_viz_snapshot <src.tracking.logging.modules.policy_telemetry_db.persist_policy_viz_snapshot>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.persist_policy_viz_snapshot
    :summary:
    ```
* - {py:obj}`_x_axis_from_viz <src.tracking.logging.modules.policy_telemetry_db._x_axis_from_viz>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._x_axis_from_viz
    :summary:
    ```
* - {py:obj}`_metric_series <src.tracking.logging.modules.policy_telemetry_db._metric_series>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._metric_series
    :summary:
    ```
* - {py:obj}`query_policy_trajectory_series <src.tracking.logging.modules.policy_telemetry_db.query_policy_trajectory_series>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.query_policy_trajectory_series
    :summary:
    ```
* - {py:obj}`query_policy_telemetry_trends <src.tracking.logging.modules.policy_telemetry_db.query_policy_telemetry_trends>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.query_policy_telemetry_trends
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DB_LOCK <src.tracking.logging.modules.policy_telemetry_db._DB_LOCK>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._DB_LOCK
    :summary:
    ```
* - {py:obj}`_SCHEMA_VERSION <src.tracking.logging.modules.policy_telemetry_db._SCHEMA_VERSION>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._SCHEMA_VERSION
    :summary:
    ```
* - {py:obj}`base_uri <src.tracking.logging.modules.policy_telemetry_db.base_uri>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.base_uri
    :summary:
    ```
* - {py:obj}`TELEMETRY_DB_PATH <src.tracking.logging.modules.policy_telemetry_db.TELEMETRY_DB_PATH>`
  - ```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.TELEMETRY_DB_PATH
    :summary:
    ```
````

### API

````{py:data} _DB_LOCK
:canonical: src.tracking.logging.modules.policy_telemetry_db._DB_LOCK
:value: >
   'Lock(...)'

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._DB_LOCK
```

````

````{py:data} _SCHEMA_VERSION
:canonical: src.tracking.logging.modules.policy_telemetry_db._SCHEMA_VERSION
:value: >
   1

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._SCHEMA_VERSION
```

````

````{py:data} base_uri
:canonical: src.tracking.logging.modules.policy_telemetry_db.base_uri
:value: >
   None

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.base_uri
```

````

````{py:data} TELEMETRY_DB_PATH
:canonical: src.tracking.logging.modules.policy_telemetry_db.TELEMETRY_DB_PATH
:type: str
:value: >
   'str(...)'

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.TELEMETRY_DB_PATH
```

````

````{py:function} _conn(timeout: float = 5.0) -> sqlite3.Connection
:canonical: src.tracking.logging.modules.policy_telemetry_db._conn

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._conn
```
````

````{py:function} _ensure_schema(conn: sqlite3.Connection) -> None
:canonical: src.tracking.logging.modules.policy_telemetry_db._ensure_schema

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._ensure_schema
```
````

````{py:function} _series_tail(values: typing.Any) -> typing.Optional[float]
:canonical: src.tracking.logging.modules.policy_telemetry_db._series_tail

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._series_tail
```
````

````{py:function} extract_final_metric(policy_type: str, viz_data: typing.Dict[str, typing.List[typing.Any]]) -> typing.Tuple[typing.Optional[float], typing.Optional[str]]
:canonical: src.tracking.logging.modules.policy_telemetry_db.extract_final_metric

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.extract_final_metric
```
````

````{py:function} _step_count(viz_data: typing.Dict[str, typing.List[typing.Any]]) -> int
:canonical: src.tracking.logging.modules.policy_telemetry_db._step_count

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._step_count
```
````

````{py:function} _run_label_from_path(log_path: typing.Optional[str]) -> typing.Optional[str]
:canonical: src.tracking.logging.modules.policy_telemetry_db._run_label_from_path

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._run_label_from_path
```
````

````{py:function} _upsert_run(conn: sqlite3.Connection, log_path: str) -> int
:canonical: src.tracking.logging.modules.policy_telemetry_db._upsert_run

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._upsert_run
```
````

````{py:function} persist_policy_viz_snapshot(viz_data: typing.Dict[str, typing.List[typing.Any]], policy: str, sample_idx: int, day: int, policy_type: str, log_path: typing.Optional[str]) -> bool
:canonical: src.tracking.logging.modules.policy_telemetry_db.persist_policy_viz_snapshot

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.persist_policy_viz_snapshot
```
````

````{py:function} _x_axis_from_viz(viz_data: typing.Dict[str, typing.List[typing.Any]]) -> typing.List[int]
:canonical: src.tracking.logging.modules.policy_telemetry_db._x_axis_from_viz

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._x_axis_from_viz
```
````

````{py:function} _metric_series(policy_type: str, viz_data: typing.Dict[str, typing.List[typing.Any]]) -> typing.Tuple[typing.Optional[str], typing.List[float]]
:canonical: src.tracking.logging.modules.policy_telemetry_db._metric_series

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db._metric_series
```
````

````{py:function} query_policy_trajectory_series(policy: typing.Optional[str] = None, policy_type: typing.Optional[str] = None, run_label: typing.Optional[str] = None, limit: int = 12) -> typing.Dict[str, typing.Any]
:canonical: src.tracking.logging.modules.policy_telemetry_db.query_policy_trajectory_series

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.query_policy_trajectory_series
```
````

````{py:function} query_policy_telemetry_trends(policy_type: typing.Optional[str] = None, run_label: typing.Optional[str] = None, limit: int = 500) -> typing.Dict[str, typing.Any]
:canonical: src.tracking.logging.modules.policy_telemetry_db.query_policy_telemetry_trends

```{autodoc2-docstring} src.tracking.logging.modules.policy_telemetry_db.query_policy_telemetry_trends
```
````
