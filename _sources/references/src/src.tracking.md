# {py:mod}`src.tracking`

```{py:module} src.tracking
```

```{autodoc2-docstring} src.tracking
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.tracking.logging
src.tracking.database
src.tracking.core
src.tracking.integrations
src.tracking.hooks
src.tracking.profiling
src.tracking.helpers
src.tracking.validation
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.tracking.viz_mixin
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`init <src.tracking.init>`
  - ```{autodoc2-docstring} src.tracking.init
    :summary:
    ```
* - {py:obj}`init_worker <src.tracking.init_worker>`
  - ```{autodoc2-docstring} src.tracking.init_worker
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.tracking.__all__>`
  - ```{autodoc2-docstring} src.tracking.__all__
    :summary:
    ```
* - {py:obj}`_DEFAULT_TRACKING_URI <src.tracking._DEFAULT_TRACKING_URI>`
  - ```{autodoc2-docstring} src.tracking._DEFAULT_TRACKING_URI
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.tracking.__all__
:value: >
   ['init', 'init_worker', 'get_tracker', 'get_active_run', 'Tracker', 'Run', 'TrackingCallback', 'Simu...

```{autodoc2-docstring} src.tracking.__all__
```

````

````{py:data} _DEFAULT_TRACKING_URI
:canonical: src.tracking._DEFAULT_TRACKING_URI
:value: >
   'assets/tracking'

```{autodoc2-docstring} src.tracking._DEFAULT_TRACKING_URI
```

````

````{py:function} init(experiment_name: str, tracking_uri: typing.Optional[str] = None, run_type: str = 'generic', tags: typing.Optional[typing.Dict[str, str]] = None, description: str = '', buffer_size: int = 200) -> src.tracking.core.tracker.Tracker
:canonical: src.tracking.init

```{autodoc2-docstring} src.tracking.init
```
````

````{py:function} init_worker(tracking_uri: str, run_id: str, buffer_size: int = 200) -> typing.Optional[src.tracking.core.run.Run]
:canonical: src.tracking.init_worker

```{autodoc2-docstring} src.tracking.init_worker
```
````
