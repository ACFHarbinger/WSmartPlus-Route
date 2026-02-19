# {py:mod}`src.pipeline.ui.pages.simulation_summary`

```{py:module} src.pipeline.ui.pages.simulation_summary
```

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_discover_output_dirs <src.pipeline.ui.pages.simulation_summary._discover_output_dirs>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._discover_output_dirs
    :summary:
    ```
* - {py:obj}`_load_json <src.pipeline.ui.pages.simulation_summary._load_json>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._load_json
    :summary:
    ```
* - {py:obj}`_find_json_files <src.pipeline.ui.pages.simulation_summary._find_json_files>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._find_json_files
    :summary:
    ```
* - {py:obj}`_parse_policy_name <src.pipeline.ui.pages.simulation_summary._parse_policy_name>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._parse_policy_name
    :summary:
    ```
* - {py:obj}`_extract_distributions <src.pipeline.ui.pages.simulation_summary._extract_distributions>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._extract_distributions
    :summary:
    ```
* - {py:obj}`_build_summary_df <src.pipeline.ui.pages.simulation_summary._build_summary_df>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._build_summary_df
    :summary:
    ```
* - {py:obj}`_build_daily_df <src.pipeline.ui.pages.simulation_summary._build_daily_df>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._build_daily_df
    :summary:
    ```
* - {py:obj}`_filter_by_dist <src.pipeline.ui.pages.simulation_summary._filter_by_dist>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._filter_by_dist
    :summary:
    ```
* - {py:obj}`_render_kpi_overview <src.pipeline.ui.pages.simulation_summary._render_kpi_overview>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_kpi_overview
    :summary:
    ```
* - {py:obj}`_render_summary_table <src.pipeline.ui.pages.simulation_summary._render_summary_table>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_summary_table
    :summary:
    ```
* - {py:obj}`_render_metric_bar_chart <src.pipeline.ui.pages.simulation_summary._render_metric_bar_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_metric_bar_chart
    :summary:
    ```
* - {py:obj}`_render_pareto <src.pipeline.ui.pages.simulation_summary._render_pareto>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_pareto
    :summary:
    ```
* - {py:obj}`_render_distribution_comparison <src.pipeline.ui.pages.simulation_summary._render_distribution_comparison>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_distribution_comparison
    :summary:
    ```
* - {py:obj}`_render_daily_timeseries <src.pipeline.ui.pages.simulation_summary._render_daily_timeseries>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_daily_timeseries
    :summary:
    ```
* - {py:obj}`_render_sidebar_controls <src.pipeline.ui.pages.simulation_summary._render_sidebar_controls>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_sidebar_controls
    :summary:
    ```
* - {py:obj}`render_simulation_summary <src.pipeline.ui.pages.simulation_summary.render_simulation_summary>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary.render_simulation_summary
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DISPLAY_METRICS <src.pipeline.ui.pages.simulation_summary._DISPLAY_METRICS>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._DISPLAY_METRICS
    :summary:
    ```
* - {py:obj}`_DIST_PATTERN <src.pipeline.ui.pages.simulation_summary._DIST_PATTERN>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._DIST_PATTERN
    :summary:
    ```
````

### API

````{py:data} _DISPLAY_METRICS
:canonical: src.pipeline.ui.pages.simulation_summary._DISPLAY_METRICS
:value: >
   ['profit', 'cost', 'kg', 'km', 'kg/km', 'overflows', 'ncol', 'kg_lost', 'days', 'time']

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._DISPLAY_METRICS
```

````

````{py:data} _DIST_PATTERN
:canonical: src.pipeline.ui.pages.simulation_summary._DIST_PATTERN
:value: >
   'compile(...)'

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._DIST_PATTERN
```

````

````{py:function} _discover_output_dirs() -> typing.List[str]
:canonical: src.pipeline.ui.pages.simulation_summary._discover_output_dirs

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._discover_output_dirs
```
````

````{py:function} _load_json(path: str) -> typing.Any
:canonical: src.pipeline.ui.pages.simulation_summary._load_json

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._load_json
```
````

````{py:function} _find_json_files(output_dir: str) -> typing.Dict[str, str]
:canonical: src.pipeline.ui.pages.simulation_summary._find_json_files

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._find_json_files
```
````

````{py:function} _parse_policy_name(raw_name: str) -> typing.Tuple[str, str]
:canonical: src.pipeline.ui.pages.simulation_summary._parse_policy_name

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._parse_policy_name
```
````

````{py:function} _extract_distributions(mean_data: typing.Dict[str, typing.Any]) -> typing.List[str]
:canonical: src.pipeline.ui.pages.simulation_summary._extract_distributions

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._extract_distributions
```
````

````{py:function} _build_summary_df(mean_data: typing.Dict[str, typing.Dict[str, float]], std_data: typing.Optional[typing.Dict[str, typing.Dict[str, float]]] = None) -> pandas.DataFrame
:canonical: src.pipeline.ui.pages.simulation_summary._build_summary_df

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._build_summary_df
```
````

````{py:function} _build_daily_df(daily_data: typing.Dict[str, typing.Dict[str, typing.Any]]) -> pandas.DataFrame
:canonical: src.pipeline.ui.pages.simulation_summary._build_daily_df

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._build_daily_df
```
````

````{py:function} _filter_by_dist(df: pandas.DataFrame, dist_filter: str) -> pandas.DataFrame
:canonical: src.pipeline.ui.pages.simulation_summary._filter_by_dist

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._filter_by_dist
```
````

````{py:function} _render_kpi_overview(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.pipeline.ui.pages.simulation_summary._render_kpi_overview

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_kpi_overview
```
````

````{py:function} _render_summary_table(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.pipeline.ui.pages.simulation_summary._render_summary_table

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_summary_table
```
````

````{py:function} _render_metric_bar_chart(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.pipeline.ui.pages.simulation_summary._render_metric_bar_chart

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_metric_bar_chart
```
````

````{py:function} _render_pareto(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.pipeline.ui.pages.simulation_summary._render_pareto

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_pareto
```
````

````{py:function} _render_distribution_comparison(summary_df: pandas.DataFrame) -> None
:canonical: src.pipeline.ui.pages.simulation_summary._render_distribution_comparison

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_distribution_comparison
```
````

````{py:function} _render_daily_timeseries(daily_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.pipeline.ui.pages.simulation_summary._render_daily_timeseries

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_daily_timeseries
```
````

````{py:function} _render_sidebar_controls(available_dirs: typing.List[str], distributions: typing.List[str]) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.ui.pages.simulation_summary._render_sidebar_controls

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary._render_sidebar_controls
```
````

````{py:function} render_simulation_summary() -> None
:canonical: src.pipeline.ui.pages.simulation_summary.render_simulation_summary

```{autodoc2-docstring} src.pipeline.ui.pages.simulation_summary.render_simulation_summary
```
````
