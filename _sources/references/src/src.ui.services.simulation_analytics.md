# {py:mod}`src.ui.services.simulation_analytics`

```{py:module} src.ui.services.simulation_analytics
```

```{autodoc2-docstring} src.ui.services.simulation_analytics
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_filter_entries <src.ui.services.simulation_analytics._filter_entries>`
  - ```{autodoc2-docstring} src.ui.services.simulation_analytics._filter_entries
    :summary:
    ```
* - {py:obj}`compute_cumulative_stats <src.ui.services.simulation_analytics.compute_cumulative_stats>`
  - ```{autodoc2-docstring} src.ui.services.simulation_analytics.compute_cumulative_stats
    :summary:
    ```
* - {py:obj}`compute_day_deltas <src.ui.services.simulation_analytics.compute_day_deltas>`
  - ```{autodoc2-docstring} src.ui.services.simulation_analytics.compute_day_deltas
    :summary:
    ```
* - {py:obj}`compute_summary_statistics <src.ui.services.simulation_analytics.compute_summary_statistics>`
  - ```{autodoc2-docstring} src.ui.services.simulation_analytics.compute_summary_statistics
    :summary:
    ```
* - {py:obj}`get_metric_history <src.ui.services.simulation_analytics.get_metric_history>`
  - ```{autodoc2-docstring} src.ui.services.simulation_analytics.get_metric_history
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_METRIC_KEYS <src.ui.services.simulation_analytics._METRIC_KEYS>`
  - ```{autodoc2-docstring} src.ui.services.simulation_analytics._METRIC_KEYS
    :summary:
    ```
````

### API

````{py:data} _METRIC_KEYS
:canonical: src.ui.services.simulation_analytics._METRIC_KEYS
:value: >
   ['profit', 'km', 'kg', 'overflows', 'cost', 'ncol', 'kg_lost', 'kg/km']

```{autodoc2-docstring} src.ui.services.simulation_analytics._METRIC_KEYS
```

````

````{py:function} _filter_entries(entries: typing.List[logic.src.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> typing.List[logic.src.ui.services.log_parser.DayLogEntry]
:canonical: src.ui.services.simulation_analytics._filter_entries

```{autodoc2-docstring} src.ui.services.simulation_analytics._filter_entries
```
````

````{py:function} compute_cumulative_stats(entries: typing.List[logic.src.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> typing.Dict[str, float]
:canonical: src.ui.services.simulation_analytics.compute_cumulative_stats

```{autodoc2-docstring} src.ui.services.simulation_analytics.compute_cumulative_stats
```
````

````{py:function} compute_day_deltas(entries: typing.List[logic.src.ui.services.log_parser.DayLogEntry], current_day: int, policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> typing.Dict[str, typing.Optional[float]]
:canonical: src.ui.services.simulation_analytics.compute_day_deltas

```{autodoc2-docstring} src.ui.services.simulation_analytics.compute_day_deltas
```
````

````{py:function} compute_summary_statistics(entries: typing.List[logic.src.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None) -> typing.Dict[str, typing.Dict[str, float]]
:canonical: src.ui.services.simulation_analytics.compute_summary_statistics

```{autodoc2-docstring} src.ui.services.simulation_analytics.compute_summary_statistics
```
````

````{py:function} get_metric_history(entries: typing.List[logic.src.ui.services.log_parser.DayLogEntry], metric: str, policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None, last_n_days: int = 7) -> typing.List[float]
:canonical: src.ui.services.simulation_analytics.get_metric_history

```{autodoc2-docstring} src.ui.services.simulation_analytics.get_metric_history
```
````
