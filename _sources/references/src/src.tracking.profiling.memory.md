# {py:mod}`src.tracking.profiling.memory`

```{py:module} src.tracking.profiling.memory
```

```{autodoc2-docstring} src.tracking.profiling.memory
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemorySnapshot <src.tracking.profiling.memory.MemorySnapshot>`
  - ```{autodoc2-docstring} src.tracking.profiling.memory.MemorySnapshot
    :summary:
    ```
* - {py:obj}`MemoryTracker <src.tracking.profiling.memory.MemoryTracker>`
  - ```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker
    :summary:
    ```
````

### API

`````{py:class} MemorySnapshot(tag: str, gpu_allocated_mb: float = 0.0, gpu_reserved_mb: float = 0.0, gpu_peak_mb: float = 0.0, cpu_rss_mb: float = 0.0, timestamp: typing.Optional[float] = None)
:canonical: src.tracking.profiling.memory.MemorySnapshot

```{autodoc2-docstring} src.tracking.profiling.memory.MemorySnapshot
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.profiling.memory.MemorySnapshot.__init__
```

````{py:method} capture(tag: str, device: typing.Optional[torch.device] = None, step: int = 0, log_metric: bool = True) -> src.tracking.profiling.memory.MemorySnapshot
:canonical: src.tracking.profiling.memory.MemorySnapshot.capture
:classmethod:

```{autodoc2-docstring} src.tracking.profiling.memory.MemorySnapshot.capture
```

````

````{py:method} delta(baseline: src.tracking.profiling.memory.MemorySnapshot) -> typing.Dict[str, float]
:canonical: src.tracking.profiling.memory.MemorySnapshot.delta

```{autodoc2-docstring} src.tracking.profiling.memory.MemorySnapshot.delta
```

````

````{py:method} log_to_run(step: int = 0) -> None
:canonical: src.tracking.profiling.memory.MemorySnapshot.log_to_run

```{autodoc2-docstring} src.tracking.profiling.memory.MemorySnapshot.log_to_run
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.profiling.memory.MemorySnapshot.__repr__

```{autodoc2-docstring} src.tracking.profiling.memory.MemorySnapshot.__repr__
```

````

`````

`````{py:class} MemoryTracker(interval_sec: float = 1.0, tag: str = 'background', device: typing.Optional[torch.device] = None, log_per_sample: bool = False)
:canonical: src.tracking.profiling.memory.MemoryTracker

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.__init__
```

````{py:property} peak_gpu_mb
:canonical: src.tracking.profiling.memory.MemoryTracker.peak_gpu_mb
:type: float

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.peak_gpu_mb
```

````

````{py:property} peak_cpu_mb
:canonical: src.tracking.profiling.memory.MemoryTracker.peak_cpu_mb
:type: float

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.peak_cpu_mb
```

````

````{py:property} n_samples
:canonical: src.tracking.profiling.memory.MemoryTracker.n_samples
:type: int

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.n_samples
```

````

````{py:method} start() -> src.tracking.profiling.memory.MemoryTracker
:canonical: src.tracking.profiling.memory.MemoryTracker.start

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.start
```

````

````{py:method} stop() -> src.tracking.profiling.memory.MemoryTracker
:canonical: src.tracking.profiling.memory.MemoryTracker.stop

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.stop
```

````

````{py:method} _monitor() -> None
:canonical: src.tracking.profiling.memory.MemoryTracker._monitor

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker._monitor
```

````

````{py:method} log_summary_to_run(step: int = 0) -> None
:canonical: src.tracking.profiling.memory.MemoryTracker.log_summary_to_run

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.log_summary_to_run
```

````

````{py:method} __enter__() -> src.tracking.profiling.memory.MemoryTracker
:canonical: src.tracking.profiling.memory.MemoryTracker.__enter__

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.__enter__
```

````

````{py:method} __exit__(*args: typing.Any) -> None
:canonical: src.tracking.profiling.memory.MemoryTracker.__exit__

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.__exit__
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.profiling.memory.MemoryTracker.__repr__

```{autodoc2-docstring} src.tracking.profiling.memory.MemoryTracker.__repr__
```

````

`````
