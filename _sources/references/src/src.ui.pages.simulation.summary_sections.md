# {py:mod}`src.ui.pages.simulation.summary_sections`

```{py:module} src.ui.pages.simulation.summary_sections
```

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_filter_by_dist <src.ui.pages.simulation.summary_sections._filter_by_dist>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._filter_by_dist
    :summary:
    ```
* - {py:obj}`_render_kpi_overview <src.ui.pages.simulation.summary_sections._render_kpi_overview>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_kpi_overview
    :summary:
    ```
* - {py:obj}`_render_summary_table <src.ui.pages.simulation.summary_sections._render_summary_table>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_summary_table
    :summary:
    ```
* - {py:obj}`_render_metric_bar_chart <src.ui.pages.simulation.summary_sections._render_metric_bar_chart>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_metric_bar_chart
    :summary:
    ```
* - {py:obj}`_render_pareto <src.ui.pages.simulation.summary_sections._render_pareto>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_pareto
    :summary:
    ```
* - {py:obj}`_render_distribution_comparison <src.ui.pages.simulation.summary_sections._render_distribution_comparison>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_distribution_comparison
    :summary:
    ```
* - {py:obj}`_render_daily_timeseries <src.ui.pages.simulation.summary_sections._render_daily_timeseries>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_daily_timeseries
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DISPLAY_METRICS <src.ui.pages.simulation.summary_sections._DISPLAY_METRICS>`
  - ```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._DISPLAY_METRICS
    :summary:
    ```
````

### API

````{py:data} _DISPLAY_METRICS
:canonical: src.ui.pages.simulation.summary_sections._DISPLAY_METRICS
:value: >
   ['profit', 'cost', 'kg', 'km', 'kg/km', 'overflows', 'ncol', 'kg_lost', 'days', 'time']

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._DISPLAY_METRICS
```

````

````{py:function} _filter_by_dist(df: pandas.DataFrame, dist_filter: str) -> pandas.DataFrame
:canonical: src.ui.pages.simulation.summary_sections._filter_by_dist

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._filter_by_dist
```
````

````{py:function} _render_kpi_overview(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.ui.pages.simulation.summary_sections._render_kpi_overview

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_kpi_overview
```
````

````{py:function} _render_summary_table(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.ui.pages.simulation.summary_sections._render_summary_table

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_summary_table
```
````

````{py:function} _render_metric_bar_chart(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.ui.pages.simulation.summary_sections._render_metric_bar_chart

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_metric_bar_chart
```
````

````{py:function} _render_pareto(summary_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.ui.pages.simulation.summary_sections._render_pareto

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_pareto
```
````

````{py:function} _render_distribution_comparison(summary_df: pandas.DataFrame) -> None
:canonical: src.ui.pages.simulation.summary_sections._render_distribution_comparison

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_distribution_comparison
```
````

````{py:function} _render_daily_timeseries(daily_df: pandas.DataFrame, dist_filter: str) -> None
:canonical: src.ui.pages.simulation.summary_sections._render_daily_timeseries

```{autodoc2-docstring} src.ui.pages.simulation.summary_sections._render_daily_timeseries
```
````
