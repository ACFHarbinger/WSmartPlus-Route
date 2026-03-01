# {py:mod}`src.ui.components.alns_charts`

```{py:module} src.ui.components.alns_charts
```

```{autodoc2-docstring} src.ui.components.alns_charts
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrackedVectorizedALNS <src.ui.components.alns_charts.TrackedVectorizedALNS>`
  - ```{autodoc2-docstring} src.ui.components.alns_charts.TrackedVectorizedALNS
    :summary:
    ```
* - {py:obj}`ALNSSnapshotTracker <src.ui.components.alns_charts.ALNSSnapshotTracker>`
  - ```{autodoc2-docstring} src.ui.components.alns_charts.ALNSSnapshotTracker
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_alns_operator_charts <src.ui.components.alns_charts.render_alns_operator_charts>`
  - ```{autodoc2-docstring} src.ui.components.alns_charts.render_alns_operator_charts
    :summary:
    ```
* - {py:obj}`_build_chart <src.ui.components.alns_charts._build_chart>`
  - ```{autodoc2-docstring} src.ui.components.alns_charts._build_chart
    :summary:
    ```
* - {py:obj}`_render_prob_table <src.ui.components.alns_charts._render_prob_table>`
  - ```{autodoc2-docstring} src.ui.components.alns_charts._render_prob_table
    :summary:
    ```
* - {py:obj}`_softmax_probs <src.ui.components.alns_charts._softmax_probs>`
  - ```{autodoc2-docstring} src.ui.components.alns_charts._softmax_probs
    :summary:
    ```
````

### API

`````{py:class} TrackedVectorizedALNS(*args: typing.Any, log_freq: int = 25, **kwargs: typing.Any)
:canonical: src.ui.components.alns_charts.TrackedVectorizedALNS

```{autodoc2-docstring} src.ui.components.alns_charts.TrackedVectorizedALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.ui.components.alns_charts.TrackedVectorizedALNS.__init__
```

````{py:method} __getattr__(name: str) -> typing.Any
:canonical: src.ui.components.alns_charts.TrackedVectorizedALNS.__getattr__

```{autodoc2-docstring} src.ui.components.alns_charts.TrackedVectorizedALNS.__getattr__
```

````

````{py:method} solve(initial_solutions: typing.Any, n_iterations: int = 2000, time_limit: typing.Optional[float] = None, max_vehicles: int = 0, start_temp: float = 0.5, cooling_rate: float = 0.9995, **kwargs: typing.Any) -> typing.Any
:canonical: src.ui.components.alns_charts.TrackedVectorizedALNS.solve

```{autodoc2-docstring} src.ui.components.alns_charts.TrackedVectorizedALNS.solve
```

````

`````

`````{py:class} ALNSSnapshotTracker()
:canonical: src.ui.components.alns_charts.ALNSSnapshotTracker

```{autodoc2-docstring} src.ui.components.alns_charts.ALNSSnapshotTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.ui.components.alns_charts.ALNSSnapshotTracker.__init__
```

````{py:method} before(solver: typing.Any) -> None
:canonical: src.ui.components.alns_charts.ALNSSnapshotTracker.before

```{autodoc2-docstring} src.ui.components.alns_charts.ALNSSnapshotTracker.before
```

````

````{py:method} after(solver: typing.Any) -> None
:canonical: src.ui.components.alns_charts.ALNSSnapshotTracker.after

```{autodoc2-docstring} src.ui.components.alns_charts.ALNSSnapshotTracker.after
```

````

````{py:method} _snapshot(solver: typing.Any, label: str) -> None
:canonical: src.ui.components.alns_charts.ALNSSnapshotTracker._snapshot

```{autodoc2-docstring} src.ui.components.alns_charts.ALNSSnapshotTracker._snapshot
```

````

`````

````{py:function} render_alns_operator_charts(weight_history: typing.Dict[str, typing.List[typing.List[float]]], destroy_op_names: typing.Optional[typing.List[str]] = None, repair_op_names: typing.Optional[typing.List[str]] = None, chart_type: str = 'line', smooth_window: int = 1, height: int = 400, title: str = 'ALNS Operator Weight Dynamics', color_palette: typing.Optional[typing.List[str]] = None) -> None
:canonical: src.ui.components.alns_charts.render_alns_operator_charts

```{autodoc2-docstring} src.ui.components.alns_charts.render_alns_operator_charts
```
````

````{py:function} _build_chart(data: typing.List[typing.List[float]], names: typing.List[str], chart_title: str, chart_type: str, smooth_window: int, height: int, colors: typing.List[str]) -> typing.Optional[typing.Any]
:canonical: src.ui.components.alns_charts._build_chart

```{autodoc2-docstring} src.ui.components.alns_charts._build_chart
```
````

````{py:function} _render_prob_table(destroy_data: typing.List[typing.List[float]], d_names: typing.List[str], repair_data: typing.List[typing.List[float]], r_names: typing.List[str]) -> None
:canonical: src.ui.components.alns_charts._render_prob_table

```{autodoc2-docstring} src.ui.components.alns_charts._render_prob_table
```
````

````{py:function} _softmax_probs(weights: typing.Any) -> typing.List[float]
:canonical: src.ui.components.alns_charts._softmax_probs

```{autodoc2-docstring} src.ui.components.alns_charts._softmax_probs
```
````
