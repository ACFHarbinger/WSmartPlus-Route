# {py:mod}`src.tracking.core.tracker`

```{py:module} src.tracking.core.tracker
```

```{autodoc2-docstring} src.tracking.core.tracker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Tracker <src.tracking.core.tracker.Tracker>`
  - ```{autodoc2-docstring} src.tracking.core.tracker.Tracker
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_tracker <src.tracking.core.tracker.get_tracker>`
  - ```{autodoc2-docstring} src.tracking.core.tracker.get_tracker
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_tracker <src.tracking.core.tracker._tracker>`
  - ```{autodoc2-docstring} src.tracking.core.tracker._tracker
    :summary:
    ```
````

### API

````{py:data} _tracker
:canonical: src.tracking.core.tracker._tracker
:type: typing.Optional[src.tracking.core.tracker.Tracker]
:value: >
   None

```{autodoc2-docstring} src.tracking.core.tracker._tracker
```

````

````{py:function} get_tracker() -> typing.Optional[Tracker]
:canonical: src.tracking.core.tracker.get_tracker

```{autodoc2-docstring} src.tracking.core.tracker.get_tracker
```
````

`````{py:class} Tracker(tracking_uri: str, buffer_size: int = 200)
:canonical: src.tracking.core.tracker.Tracker

```{autodoc2-docstring} src.tracking.core.tracker.Tracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.core.tracker.Tracker.__init__
```

````{py:method} start_run(experiment_name: str, run_name: typing.Optional[str] = None, run_type: str = 'generic', tags: typing.Optional[typing.Dict[str, str]] = None, description: str = '') -> src.tracking.core.run.Run
:canonical: src.tracking.core.tracker.Tracker.start_run

```{autodoc2-docstring} src.tracking.core.tracker.Tracker.start_run
```

````

````{py:method} attach_run(run_id: str) -> src.tracking.core.run.Run
:canonical: src.tracking.core.tracker.Tracker.attach_run

```{autodoc2-docstring} src.tracking.core.tracker.Tracker.attach_run
```

````

````{py:method} list_experiments() -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.tracker.Tracker.list_experiments

```{autodoc2-docstring} src.tracking.core.tracker.Tracker.list_experiments
```

````

````{py:method} list_runs(experiment_name: typing.Optional[str] = None, run_type: typing.Optional[str] = None, status: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.tracker.Tracker.list_runs

```{autodoc2-docstring} src.tracking.core.tracker.Tracker.list_runs
```

````

````{py:method} get_run(run_id: str) -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.tracking.core.tracker.Tracker.get_run

```{autodoc2-docstring} src.tracking.core.tracker.Tracker.get_run
```

````

`````
