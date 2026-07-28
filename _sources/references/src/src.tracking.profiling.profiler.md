# {py:mod}`src.tracking.profiling.profiler`

```{py:module} src.tracking.profiling.profiler
```

```{autodoc2-docstring} src.tracking.profiling.profiler
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExecutionProfiler <src.tracking.profiling.profiler.ExecutionProfiler>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`start_global_profiling <src.tracking.profiling.profiler.start_global_profiling>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler.start_global_profiling
    :summary:
    ```
* - {py:obj}`stop_global_profiling <src.tracking.profiling.profiler.stop_global_profiling>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler.stop_global_profiling
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.tracking.profiling.profiler.logger>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler.logger
    :summary:
    ```
* - {py:obj}`_ROOT_DIR <src.tracking.profiling.profiler._ROOT_DIR>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler._ROOT_DIR
    :summary:
    ```
* - {py:obj}`_BUFFER_SIZE <src.tracking.profiling.profiler._BUFFER_SIZE>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler._BUFFER_SIZE
    :summary:
    ```
* - {py:obj}`_LIB_DIRS <src.tracking.profiling.profiler._LIB_DIRS>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler._LIB_DIRS
    :summary:
    ```
* - {py:obj}`_profiler_instance <src.tracking.profiling.profiler._profiler_instance>`
  - ```{autodoc2-docstring} src.tracking.profiling.profiler._profiler_instance
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.tracking.profiling.profiler.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.tracking.profiling.profiler.logger
```

````

````{py:data} _ROOT_DIR
:canonical: src.tracking.profiling.profiler._ROOT_DIR
:value: >
   'str(...)'

```{autodoc2-docstring} src.tracking.profiling.profiler._ROOT_DIR
```

````

````{py:data} _BUFFER_SIZE
:canonical: src.tracking.profiling.profiler._BUFFER_SIZE
:value: >
   200

```{autodoc2-docstring} src.tracking.profiling.profiler._BUFFER_SIZE
```

````

````{py:data} _LIB_DIRS
:canonical: src.tracking.profiling.profiler._LIB_DIRS
:value: >
   ('site-packages', 'dist-packages', 'lib/python', '<frozen', '<built-in', 'logic/src/tracking/')

```{autodoc2-docstring} src.tracking.profiling.profiler._LIB_DIRS
```

````

`````{py:class} ExecutionProfiler(log_dir: str = 'logs', buffer_size: int = _BUFFER_SIZE)
:canonical: src.tracking.profiling.profiler.ExecutionProfiler

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler.__init__
```

````{py:method} _should_profile(filename: str) -> bool
:canonical: src.tracking.profiling.profiler.ExecutionProfiler._should_profile

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler._should_profile
```

````

````{py:method} _get_class_name(frame: typing.Any) -> str
:canonical: src.tracking.profiling.profiler.ExecutionProfiler._get_class_name

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler._get_class_name
```

````

````{py:method} profile_hook(frame: typing.Any, event: str, arg: typing.Any) -> None
:canonical: src.tracking.profiling.profiler.ExecutionProfiler.profile_hook

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler.profile_hook
```

````

````{py:method} _handle_return(frame: typing.Any) -> None
:canonical: src.tracking.profiling.profiler.ExecutionProfiler._handle_return

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler._handle_return
```

````

````{py:method} _flush_locked() -> None
:canonical: src.tracking.profiling.profiler.ExecutionProfiler._flush_locked

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler._flush_locked
```

````

````{py:method} flush() -> None
:canonical: src.tracking.profiling.profiler.ExecutionProfiler.flush

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler.flush
```

````

````{py:method} start() -> None
:canonical: src.tracking.profiling.profiler.ExecutionProfiler.start

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler.start
```

````

````{py:method} stop() -> None
:canonical: src.tracking.profiling.profiler.ExecutionProfiler.stop

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler.stop
```

````

````{py:property} wall_elapsed
:canonical: src.tracking.profiling.profiler.ExecutionProfiler.wall_elapsed
:type: typing.Optional[float]

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler.wall_elapsed
```

````

````{py:method} get_report() -> typing.Any
:canonical: src.tracking.profiling.profiler.ExecutionProfiler.get_report

```{autodoc2-docstring} src.tracking.profiling.profiler.ExecutionProfiler.get_report
```

````

`````

````{py:data} _profiler_instance
:canonical: src.tracking.profiling.profiler._profiler_instance
:type: typing.Optional[src.tracking.profiling.profiler.ExecutionProfiler]
:value: >
   None

```{autodoc2-docstring} src.tracking.profiling.profiler._profiler_instance
```

````

````{py:function} start_global_profiling(log_dir: str = 'logs', buffer_size: int = _BUFFER_SIZE) -> None
:canonical: src.tracking.profiling.profiler.start_global_profiling

```{autodoc2-docstring} src.tracking.profiling.profiler.start_global_profiling
```
````

````{py:function} stop_global_profiling(log_artifact: bool = True, print_report: bool = True) -> None
:canonical: src.tracking.profiling.profiler.stop_global_profiling

```{autodoc2-docstring} src.tracking.profiling.profiler.stop_global_profiling
```
````
