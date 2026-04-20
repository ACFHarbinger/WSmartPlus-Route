# {py:mod}`src.tracking.core.run`

```{py:module} src.tracking.core.run
```

```{autodoc2-docstring} src.tracking.core.run
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Run <src.tracking.core.run.Run>`
  - ```{autodoc2-docstring} src.tracking.core.run.Run
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_active_run <src.tracking.core.run.get_active_run>`
  - ```{autodoc2-docstring} src.tracking.core.run.get_active_run
    :summary:
    ```
* - {py:obj}`set_active_run <src.tracking.core.run.set_active_run>`
  - ```{autodoc2-docstring} src.tracking.core.run.set_active_run
    :summary:
    ```
* - {py:obj}`_flatten_dict <src.tracking.core.run._flatten_dict>`
  - ```{autodoc2-docstring} src.tracking.core.run._flatten_dict
    :summary:
    ```
* - {py:obj}`_safe_float <src.tracking.core.run._safe_float>`
  - ```{autodoc2-docstring} src.tracking.core.run._safe_float
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_active_run <src.tracking.core.run._active_run>`
  - ```{autodoc2-docstring} src.tracking.core.run._active_run
    :summary:
    ```
* - {py:obj}`_registry_lock <src.tracking.core.run._registry_lock>`
  - ```{autodoc2-docstring} src.tracking.core.run._registry_lock
    :summary:
    ```
````

### API

````{py:data} _active_run
:canonical: src.tracking.core.run._active_run
:type: typing.Optional[src.tracking.core.run.Run]
:value: >
   None

```{autodoc2-docstring} src.tracking.core.run._active_run
```

````

````{py:data} _registry_lock
:canonical: src.tracking.core.run._registry_lock
:value: >
   'Lock(...)'

```{autodoc2-docstring} src.tracking.core.run._registry_lock
```

````

````{py:function} get_active_run() -> typing.Optional[Run]
:canonical: src.tracking.core.run.get_active_run

```{autodoc2-docstring} src.tracking.core.run.get_active_run
```
````

````{py:function} set_active_run(run: typing.Optional[Run]) -> None
:canonical: src.tracking.core.run.set_active_run

```{autodoc2-docstring} src.tracking.core.run.set_active_run
```
````

````{py:function} _flatten_dict(d: typing.Dict[str, typing.Any], prefix: str = '', sep: str = '.') -> typing.Dict[str, typing.Any]
:canonical: src.tracking.core.run._flatten_dict

```{autodoc2-docstring} src.tracking.core.run._flatten_dict
```
````

````{py:function} _safe_float(v: typing.Any) -> typing.Optional[float]
:canonical: src.tracking.core.run._safe_float

```{autodoc2-docstring} src.tracking.core.run._safe_float
```
````

`````{py:class} Run(run_id: str, store: logic.src.tracking.core.store.TrackingStore, artifact_dir: str, buffer_size: int = 200)
:canonical: src.tracking.core.run.Run

```{autodoc2-docstring} src.tracking.core.run.Run
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.core.run.Run.__init__
```

````{py:method} add_sink(sink: typing.Any) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.add_sink

```{autodoc2-docstring} src.tracking.core.run.Run.add_sink
```

````

````{py:method} set_tag(key: str, value: str) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.set_tag

```{autodoc2-docstring} src.tracking.core.run.Run.set_tag
```

````

````{py:method} set_tags(tags: typing.Dict[str, str]) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.set_tags

```{autodoc2-docstring} src.tracking.core.run.Run.set_tags
```

````

````{py:method} log_param(key: str, value: typing.Any) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.log_param

```{autodoc2-docstring} src.tracking.core.run.Run.log_param
```

````

````{py:method} log_params(params: typing.Dict[str, typing.Any]) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.log_params

```{autodoc2-docstring} src.tracking.core.run.Run.log_params
```

````

````{py:method} log_metric(key: str, value: typing.Any, step: int = 0) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.log_metric

```{autodoc2-docstring} src.tracking.core.run.Run.log_metric
```

````

````{py:method} log_metrics(metrics: typing.Dict[str, typing.Any], step: int = 0) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.log_metrics

```{autodoc2-docstring} src.tracking.core.run.Run.log_metrics
```

````

````{py:method} flush() -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.flush

```{autodoc2-docstring} src.tracking.core.run.Run.flush
```

````

````{py:method} _flush_metrics_locked() -> None
:canonical: src.tracking.core.run.Run._flush_metrics_locked

```{autodoc2-docstring} src.tracking.core.run.Run._flush_metrics_locked
```

````

````{py:method} log_artifact(path: str, name: typing.Optional[str] = None, artifact_type: str = 'file', metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.log_artifact

```{autodoc2-docstring} src.tracking.core.run.Run.log_artifact
```

````

````{py:method} log_artifacts_dir(dir_path: str, artifact_type: str = 'file') -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.log_artifacts_dir

```{autodoc2-docstring} src.tracking.core.run.Run.log_artifacts_dir
```

````

````{py:method} log_dataset_event(event_type: str, file_path: typing.Optional[str] = None, shape: typing.Optional[tuple] = None, metadata: typing.Optional[typing.Dict[str, typing.Any]] = None) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.log_dataset_event

```{autodoc2-docstring} src.tracking.core.run.Run.log_dataset_event
```

````

````{py:method} watch_file(file_path: str) -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.watch_file

```{autodoc2-docstring} src.tracking.core.run.Run.watch_file
```

````

````{py:method} check_file_changed(file_path: str) -> bool
:canonical: src.tracking.core.run.Run.check_file_changed

```{autodoc2-docstring} src.tracking.core.run.Run.check_file_changed
```

````

````{py:method} finish(status: str = 'completed', error: typing.Optional[str] = None) -> None
:canonical: src.tracking.core.run.Run.finish

```{autodoc2-docstring} src.tracking.core.run.Run.finish
```

````

````{py:method} __enter__() -> src.tracking.core.run.Run
:canonical: src.tracking.core.run.Run.__enter__

```{autodoc2-docstring} src.tracking.core.run.Run.__enter__
```

````

````{py:method} __exit__(exc_type: typing.Any, exc_val: typing.Any, _exc_tb: typing.Any) -> typing_extensions.Literal[False]
:canonical: src.tracking.core.run.Run.__exit__

```{autodoc2-docstring} src.tracking.core.run.Run.__exit__
```

````

````{py:method} get_params() -> typing.Dict[str, typing.Any]
:canonical: src.tracking.core.run.Run.get_params

```{autodoc2-docstring} src.tracking.core.run.Run.get_params
```

````

````{py:method} get_tags() -> typing.Dict[str, str]
:canonical: src.tracking.core.run.Run.get_tags

```{autodoc2-docstring} src.tracking.core.run.Run.get_tags
```

````

````{py:method} get_latest_metrics() -> typing.Dict[str, float]
:canonical: src.tracking.core.run.Run.get_latest_metrics

```{autodoc2-docstring} src.tracking.core.run.Run.get_latest_metrics
```

````

````{py:method} get_metric_history(key: str) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.run.Run.get_metric_history

```{autodoc2-docstring} src.tracking.core.run.Run.get_metric_history
```

````

````{py:method} get_artifacts(artifact_type: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.run.Run.get_artifacts

```{autodoc2-docstring} src.tracking.core.run.Run.get_artifacts
```

````

````{py:method} get_dataset_events() -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.run.Run.get_dataset_events

```{autodoc2-docstring} src.tracking.core.run.Run.get_dataset_events
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.core.run.Run.__repr__

````

`````
