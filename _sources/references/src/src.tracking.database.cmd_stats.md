# {py:mod}`src.tracking.database.cmd_stats`

```{py:module} src.tracking.database.cmd_stats
```

```{autodoc2-docstring} src.tracking.database.cmd_stats
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_human_bytes <src.tracking.database.cmd_stats._human_bytes>`
  - ```{autodoc2-docstring} src.tracking.database.cmd_stats._human_bytes
    :summary:
    ```
* - {py:obj}`_human_duration <src.tracking.database.cmd_stats._human_duration>`
  - ```{autodoc2-docstring} src.tracking.database.cmd_stats._human_duration
    :summary:
    ```
* - {py:obj}`_sparkbar <src.tracking.database.cmd_stats._sparkbar>`
  - ```{autodoc2-docstring} src.tracking.database.cmd_stats._sparkbar
    :summary:
    ```
* - {py:obj}`stats_database <src.tracking.database.cmd_stats.stats_database>`
  - ```{autodoc2-docstring} src.tracking.database.cmd_stats.stats_database
    :summary:
    ```
* - {py:obj}`metrics_summary <src.tracking.database.cmd_stats.metrics_summary>`
  - ```{autodoc2-docstring} src.tracking.database.cmd_stats.metrics_summary
    :summary:
    ```
````

### API

````{py:function} _human_bytes(n: int) -> str
:canonical: src.tracking.database.cmd_stats._human_bytes

```{autodoc2-docstring} src.tracking.database.cmd_stats._human_bytes
```
````

````{py:function} _human_duration(seconds: float) -> str
:canonical: src.tracking.database.cmd_stats._human_duration

```{autodoc2-docstring} src.tracking.database.cmd_stats._human_duration
```
````

````{py:function} _sparkbar(value: int, max_value: int, width: int = 20) -> str
:canonical: src.tracking.database.cmd_stats._sparkbar

```{autodoc2-docstring} src.tracking.database.cmd_stats._sparkbar
```
````

````{py:function} stats_database(experiment_name: str = '') -> None
:canonical: src.tracking.database.cmd_stats.stats_database

```{autodoc2-docstring} src.tracking.database.cmd_stats.stats_database
```
````

````{py:function} metrics_summary(key: str = '', experiment_name: str = '') -> None
:canonical: src.tracking.database.cmd_stats.metrics_summary

```{autodoc2-docstring} src.tracking.database.cmd_stats.metrics_summary
```
````
