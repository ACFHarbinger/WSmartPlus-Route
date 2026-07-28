# {py:mod}`src.tracking.profiling.throughput`

```{py:module} src.tracking.profiling.throughput
```

```{autodoc2-docstring} src.tracking.profiling.throughput
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThroughputTracker <src.tracking.profiling.throughput.ThroughputTracker>`
  - ```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker
    :summary:
    ```
````

### API

`````{py:class} ThroughputTracker(window: int = 100, unit: str = 'items')
:canonical: src.tracking.profiling.throughput.ThroughputTracker

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.__init__
```

````{py:method} start() -> src.tracking.profiling.throughput.ThroughputTracker
:canonical: src.tracking.profiling.throughput.ThroughputTracker.start

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.start
```

````

````{py:method} reset() -> src.tracking.profiling.throughput.ThroughputTracker
:canonical: src.tracking.profiling.throughput.ThroughputTracker.reset

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.reset
```

````

````{py:method} record(n_items: int = 1) -> None
:canonical: src.tracking.profiling.throughput.ThroughputTracker.record

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.record
```

````

````{py:method} step(n_items: int = 1) -> typing.Generator[None, None, None]
:canonical: src.tracking.profiling.throughput.ThroughputTracker.step

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.step
```

````

````{py:property} throughput
:canonical: src.tracking.profiling.throughput.ThroughputTracker.throughput
:type: float

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.throughput
```

````

````{py:property} total_throughput
:canonical: src.tracking.profiling.throughput.ThroughputTracker.total_throughput
:type: float

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.total_throughput
```

````

````{py:property} total_items
:canonical: src.tracking.profiling.throughput.ThroughputTracker.total_items
:type: int

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.total_items
```

````

````{py:property} elapsed
:canonical: src.tracking.profiling.throughput.ThroughputTracker.elapsed
:type: float

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.elapsed
```

````

````{py:method} summary() -> typing.Dict[str, typing.Any]
:canonical: src.tracking.profiling.throughput.ThroughputTracker.summary

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.summary
```

````

````{py:method} log_to_run(step: int = 0, prefix: str = 'throughput') -> None
:canonical: src.tracking.profiling.throughput.ThroughputTracker.log_to_run

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.log_to_run
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.profiling.throughput.ThroughputTracker.__repr__

```{autodoc2-docstring} src.tracking.profiling.throughput.ThroughputTracker.__repr__
```

````

`````
