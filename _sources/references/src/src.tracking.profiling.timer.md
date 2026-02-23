# {py:mod}`src.tracking.profiling.timer`

```{py:module} src.tracking.profiling.timer
```

```{autodoc2-docstring} src.tracking.profiling.timer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BlockTimer <src.tracking.profiling.timer.BlockTimer>`
  - ```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer
    :summary:
    ```
* - {py:obj}`MultiStepTimer <src.tracking.profiling.timer.MultiStepTimer>`
  - ```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`profile_block <src.tracking.profiling.timer.profile_block>`
  - ```{autodoc2-docstring} src.tracking.profiling.timer.profile_block
    :summary:
    ```
* - {py:obj}`profile_function <src.tracking.profiling.timer.profile_function>`
  - ```{autodoc2-docstring} src.tracking.profiling.timer.profile_function
    :summary:
    ```
````

### API

`````{py:class} BlockTimer(name: str, log_metric: bool = True, step: int = 0, prefix: str = 'time')
:canonical: src.tracking.profiling.timer.BlockTimer

```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer.__init__
```

````{py:method} start() -> src.tracking.profiling.timer.BlockTimer
:canonical: src.tracking.profiling.timer.BlockTimer.start

```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer.start
```

````

````{py:method} stop() -> float
:canonical: src.tracking.profiling.timer.BlockTimer.stop

```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer.stop
```

````

````{py:method} log_to_run() -> None
:canonical: src.tracking.profiling.timer.BlockTimer.log_to_run

```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer.log_to_run
```

````

````{py:method} __enter__() -> src.tracking.profiling.timer.BlockTimer
:canonical: src.tracking.profiling.timer.BlockTimer.__enter__

```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer.__enter__
```

````

````{py:method} __exit__(*_args: typing.Any) -> None
:canonical: src.tracking.profiling.timer.BlockTimer.__exit__

```{autodoc2-docstring} src.tracking.profiling.timer.BlockTimer.__exit__
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.profiling.timer.BlockTimer.__repr__

````

`````

`````{py:class} MultiStepTimer(accumulate: bool = True)
:canonical: src.tracking.profiling.timer.MultiStepTimer

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer.__init__
```

````{py:method} start(phase: str) -> src.tracking.profiling.timer.MultiStepTimer
:canonical: src.tracking.profiling.timer.MultiStepTimer.start

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer.start
```

````

````{py:method} stop() -> src.tracking.profiling.timer.MultiStepTimer
:canonical: src.tracking.profiling.timer.MultiStepTimer.stop

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer.stop
```

````

````{py:method} _commit() -> None
:canonical: src.tracking.profiling.timer.MultiStepTimer._commit

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer._commit
```

````

````{py:property} total
:canonical: src.tracking.profiling.timer.MultiStepTimer.total
:type: float

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer.total
```

````

````{py:method} phase_total(phase: str) -> float
:canonical: src.tracking.profiling.timer.MultiStepTimer.phase_total

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer.phase_total
```

````

````{py:method} summary() -> typing.Dict[str, float]
:canonical: src.tracking.profiling.timer.MultiStepTimer.summary

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer.summary
```

````

````{py:method} log_to_run(prefix: str = 'time', step: int = 0) -> None
:canonical: src.tracking.profiling.timer.MultiStepTimer.log_to_run

```{autodoc2-docstring} src.tracking.profiling.timer.MultiStepTimer.log_to_run
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.profiling.timer.MultiStepTimer.__repr__

````

`````

````{py:function} profile_block(name: str, log_metric: bool = True, step: int = 0, prefix: str = 'time') -> typing.Generator[src.tracking.profiling.timer.BlockTimer, None, None]
:canonical: src.tracking.profiling.timer.profile_block

```{autodoc2-docstring} src.tracking.profiling.timer.profile_block
```
````

````{py:function} profile_function(name: typing.Optional[str] = None, log_metric: bool = True, prefix: str = 'time') -> typing.Callable[..., typing.Any]
:canonical: src.tracking.profiling.timer.profile_function

```{autodoc2-docstring} src.tracking.profiling.timer.profile_function
```
````
