# {py:mod}`src.tracking.profiling.report`

```{py:module} src.tracking.profiling.report
```

```{autodoc2-docstring} src.tracking.profiling.report
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProfilingReport <src.tracking.profiling.report.ProfilingReport>`
  - ```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport
    :summary:
    ```
````

### API

`````{py:class} ProfilingReport(csv_path: str, wall_elapsed: typing.Optional[float] = None)
:canonical: src.tracking.profiling.report.ProfilingReport

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.__init__
```

````{py:method} _load() -> None
:canonical: src.tracking.profiling.report.ProfilingReport._load

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport._load
```

````

````{py:property} n_calls
:canonical: src.tracking.profiling.report.ProfilingReport.n_calls
:type: int

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.n_calls
```

````

````{py:property} total_time
:canonical: src.tracking.profiling.report.ProfilingReport.total_time
:type: float

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.total_time
```

````

````{py:method} top_functions(n: int = 20) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.profiling.report.ProfilingReport.top_functions

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.top_functions
```

````

````{py:method} file_breakdown(n: int = 15) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.profiling.report.ProfilingReport.file_breakdown

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.file_breakdown
```

````

````{py:method} class_breakdown(n: int = 15) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.profiling.report.ProfilingReport.class_breakdown

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.class_breakdown
```

````

````{py:method} module_breakdown() -> typing.List[typing.Tuple[str, float]]
:canonical: src.tracking.profiling.report.ProfilingReport.module_breakdown

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.module_breakdown
```

````

````{py:method} timeline_gaps(min_gap_sec: float = 1.0) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.tracking.profiling.report.ProfilingReport.timeline_gaps

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.timeline_gaps
```

````

````{py:method} slowest_call() -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.tracking.profiling.report.ProfilingReport.slowest_call

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.slowest_call
```

````

````{py:method} summary() -> typing.Dict[str, typing.Any]
:canonical: src.tracking.profiling.report.ProfilingReport.summary

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.summary
```

````

````{py:method} log_to_run(top_n: int = 10, step: int = 0) -> None
:canonical: src.tracking.profiling.report.ProfilingReport.log_to_run

```{autodoc2-docstring} src.tracking.profiling.report.ProfilingReport.log_to_run
```

````

````{py:method} __repr__() -> str
:canonical: src.tracking.profiling.report.ProfilingReport.__repr__

````

````{py:method} __str__() -> str
:canonical: src.tracking.profiling.report.ProfilingReport.__str__

````

`````
