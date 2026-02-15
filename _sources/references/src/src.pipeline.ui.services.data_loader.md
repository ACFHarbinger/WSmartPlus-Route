# {py:mod}`src.pipeline.ui.services.data_loader`

```{py:module} src.pipeline.ui.services.data_loader
```

```{autodoc2-docstring} src.pipeline.ui.services.data_loader
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_project_root <src.pipeline.ui.services.data_loader.get_project_root>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_project_root
    :summary:
    ```
* - {py:obj}`get_logs_dir <src.pipeline.ui.services.data_loader.get_logs_dir>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_logs_dir
    :summary:
    ```
* - {py:obj}`get_simulation_output_dir <src.pipeline.ui.services.data_loader.get_simulation_output_dir>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_simulation_output_dir
    :summary:
    ```
* - {py:obj}`discover_training_runs <src.pipeline.ui.services.data_loader.discover_training_runs>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.discover_training_runs
    :summary:
    ```
* - {py:obj}`load_hparams <src.pipeline.ui.services.data_loader.load_hparams>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_hparams
    :summary:
    ```
* - {py:obj}`load_training_metrics <src.pipeline.ui.services.data_loader.load_training_metrics>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_training_metrics
    :summary:
    ```
* - {py:obj}`load_multiple_training_runs <src.pipeline.ui.services.data_loader.load_multiple_training_runs>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_multiple_training_runs
    :summary:
    ```
* - {py:obj}`discover_simulation_logs <src.pipeline.ui.services.data_loader.discover_simulation_logs>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.discover_simulation_logs
    :summary:
    ```
* - {py:obj}`load_simulation_log <src.pipeline.ui.services.data_loader.load_simulation_log>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_simulation_log
    :summary:
    ```
* - {py:obj}`load_simulation_log_fresh <src.pipeline.ui.services.data_loader.load_simulation_log_fresh>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_simulation_log_fresh
    :summary:
    ```
* - {py:obj}`get_simulation_metadata <src.pipeline.ui.services.data_loader.get_simulation_metadata>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_simulation_metadata
    :summary:
    ```
* - {py:obj}`entries_to_dataframe <src.pipeline.ui.services.data_loader.entries_to_dataframe>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.entries_to_dataframe
    :summary:
    ```
* - {py:obj}`compute_daily_stats <src.pipeline.ui.services.data_loader.compute_daily_stats>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_daily_stats
    :summary:
    ```
* - {py:obj}`_filter_entries <src.pipeline.ui.services.data_loader._filter_entries>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader._filter_entries
    :summary:
    ```
* - {py:obj}`compute_cumulative_stats <src.pipeline.ui.services.data_loader.compute_cumulative_stats>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_cumulative_stats
    :summary:
    ```
* - {py:obj}`compute_day_deltas <src.pipeline.ui.services.data_loader.compute_day_deltas>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_day_deltas
    :summary:
    ```
* - {py:obj}`compute_summary_statistics <src.pipeline.ui.services.data_loader.compute_summary_statistics>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_summary_statistics
    :summary:
    ```
* - {py:obj}`get_metric_history <src.pipeline.ui.services.data_loader.get_metric_history>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_metric_history
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_METRIC_KEYS <src.pipeline.ui.services.data_loader._METRIC_KEYS>`
  - ```{autodoc2-docstring} src.pipeline.ui.services.data_loader._METRIC_KEYS
    :summary:
    ```
````

### API

````{py:function} get_project_root() -> pathlib.Path
:canonical: src.pipeline.ui.services.data_loader.get_project_root

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_project_root
```
````

````{py:function} get_logs_dir() -> pathlib.Path
:canonical: src.pipeline.ui.services.data_loader.get_logs_dir

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_logs_dir
```
````

````{py:function} get_simulation_output_dir() -> pathlib.Path
:canonical: src.pipeline.ui.services.data_loader.get_simulation_output_dir

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_simulation_output_dir
```
````

````{py:function} discover_training_runs() -> typing.List[typing.Tuple[str, pathlib.Path]]
:canonical: src.pipeline.ui.services.data_loader.discover_training_runs

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.discover_training_runs
```
````

````{py:function} load_hparams(version_name: str) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.ui.services.data_loader.load_hparams

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_hparams
```
````

````{py:function} load_training_metrics(metrics_path: str) -> pandas.DataFrame
:canonical: src.pipeline.ui.services.data_loader.load_training_metrics

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_training_metrics
```
````

````{py:function} load_multiple_training_runs(version_names: typing.List[str]) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.pipeline.ui.services.data_loader.load_multiple_training_runs

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_multiple_training_runs
```
````

````{py:function} discover_simulation_logs() -> typing.List[typing.Tuple[str, pathlib.Path]]
:canonical: src.pipeline.ui.services.data_loader.discover_simulation_logs

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.discover_simulation_logs
```
````

````{py:function} load_simulation_log(log_path: str) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.ui.services.data_loader.load_simulation_log

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_simulation_log
```
````

````{py:function} load_simulation_log_fresh(log_path: str) -> typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry]
:canonical: src.pipeline.ui.services.data_loader.load_simulation_log_fresh

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.load_simulation_log_fresh
```
````

````{py:function} get_simulation_metadata(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry]) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.ui.services.data_loader.get_simulation_metadata

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_simulation_metadata
```
````

````{py:function} entries_to_dataframe(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry]) -> pandas.DataFrame
:canonical: src.pipeline.ui.services.data_loader.entries_to_dataframe

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.entries_to_dataframe
```
````

````{py:function} compute_daily_stats(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.pipeline.ui.services.data_loader.compute_daily_stats

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_daily_stats
```
````

````{py:data} _METRIC_KEYS
:canonical: src.pipeline.ui.services.data_loader._METRIC_KEYS
:value: >
   ['profit', 'km', 'kg', 'overflows', 'cost', 'ncol', 'kg_lost', 'kg/km']

```{autodoc2-docstring} src.pipeline.ui.services.data_loader._METRIC_KEYS
```

````

````{py:function} _filter_entries(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry]
:canonical: src.pipeline.ui.services.data_loader._filter_entries

```{autodoc2-docstring} src.pipeline.ui.services.data_loader._filter_entries
```
````

````{py:function} compute_cumulative_stats(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> typing.Dict[str, float]
:canonical: src.pipeline.ui.services.data_loader.compute_cumulative_stats

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_cumulative_stats
```
````

````{py:function} compute_day_deltas(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], current_day: int, policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> typing.Dict[str, typing.Optional[float]]
:canonical: src.pipeline.ui.services.data_loader.compute_day_deltas

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_day_deltas
```
````

````{py:function} compute_summary_statistics(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], policy: typing.Optional[str] = None) -> typing.Dict[str, typing.Dict[str, float]]
:canonical: src.pipeline.ui.services.data_loader.compute_summary_statistics

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.compute_summary_statistics
```
````

````{py:function} get_metric_history(entries: typing.List[logic.src.pipeline.ui.services.log_parser.DayLogEntry], metric: str, policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None, last_n_days: int = 7) -> typing.List[float]
:canonical: src.pipeline.ui.services.data_loader.get_metric_history

```{autodoc2-docstring} src.pipeline.ui.services.data_loader.get_metric_history
```
````
