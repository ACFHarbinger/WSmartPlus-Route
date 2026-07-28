# {py:mod}`src.pipeline.simulations.failure_analyzer`

```{py:module} src.pipeline.simulations.failure_analyzer
```

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FailureAnalyzer <src.pipeline.simulations.failure_analyzer.FailureAnalyzer>`
  - ```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILL_SPIKE_RATIO <src.pipeline.simulations.failure_analyzer.FILL_SPIKE_RATIO>`
  - ```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FILL_SPIKE_RATIO
    :summary:
    ```
* - {py:obj}`HIGH_FILL_THRESHOLD <src.pipeline.simulations.failure_analyzer.HIGH_FILL_THRESHOLD>`
  - ```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.HIGH_FILL_THRESHOLD
    :summary:
    ```
````

### API

````{py:data} FILL_SPIKE_RATIO
:canonical: src.pipeline.simulations.failure_analyzer.FILL_SPIKE_RATIO
:value: >
   2.0

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FILL_SPIKE_RATIO
```

````

````{py:data} HIGH_FILL_THRESHOLD
:canonical: src.pipeline.simulations.failure_analyzer.HIGH_FILL_THRESHOLD
:value: >
   80.0

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.HIGH_FILL_THRESHOLD
```

````

`````{py:class} FailureAnalyzer
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer
```

````{py:method} analyze(*, new_overflows: int, sum_lost: float, profit: float, fill: numpy.ndarray, total_fill: numpy.ndarray, bins_means: numpy.ndarray, bins_real_c: numpy.ndarray, tour: typing.Sequence[int], collected: typing.Optional[numpy.ndarray], coords: typing.Union[pandas.DataFrame, typing.List[typing.Any]], mandatory: typing.Optional[typing.Sequence[int]] = None, fill_spike_ratio: float = FILL_SPIKE_RATIO, high_fill_threshold: float = HIGH_FILL_THRESHOLD) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer.analyze

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer.analyze
```

````

````{py:method} _detect_root_causes(new_overflows: int, sum_lost: float, profit: float) -> typing.List[str]
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer._detect_root_causes
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer._detect_root_causes
```

````

````{py:method} _severity(new_overflows: int, sum_lost: float, profit: float, n_overflow_bins: int) -> str
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer._severity
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer._severity
```

````

````{py:method} _summary_message(causes: typing.List[str], new_overflows: int, sum_lost: float, profit: float) -> str
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer._summary_message
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer._summary_message
```

````

````{py:method} _overflow_bins(*, fill: numpy.ndarray, bins_means: numpy.ndarray, bins_real_c: numpy.ndarray, visited: set[int], collected_arr: numpy.ndarray, coords: typing.Union[pandas.DataFrame, typing.List[typing.Any]], fill_spike_ratio: float, root_causes: typing.List[str]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer._overflow_bins

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer._overflow_bins
```

````

````{py:method} _skipped_high_fill_bins(*, total_fill: numpy.ndarray, visited: set[int], mandatory_set: set[int], coords: typing.Union[pandas.DataFrame, typing.List[typing.Any]], high_fill_threshold: float) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer._skipped_high_fill_bins

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer._skipped_high_fill_bins
```

````

````{py:method} _resolve_bin_id(bin_index: int, coords: typing.Union[pandas.DataFrame, typing.List[typing.Any]]) -> int
:canonical: src.pipeline.simulations.failure_analyzer.FailureAnalyzer._resolve_bin_id
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.failure_analyzer.FailureAnalyzer._resolve_bin_id
```

````

`````
