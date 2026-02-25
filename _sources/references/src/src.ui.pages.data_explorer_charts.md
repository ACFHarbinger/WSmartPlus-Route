# {py:mod}`src.ui.pages.data_explorer_charts`

```{py:module} src.ui.pages.data_explorer_charts
```

```{autodoc2-docstring} src.ui.pages.data_explorer_charts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_resolve_column <src.ui.pages.data_explorer_charts._resolve_column>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._resolve_column
    :summary:
    ```
* - {py:obj}`_numeric_columns <src.ui.pages.data_explorer_charts._numeric_columns>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._numeric_columns
    :summary:
    ```
* - {py:obj}`_has_distribution_meta <src.ui.pages.data_explorer_charts._has_distribution_meta>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._has_distribution_meta
    :summary:
    ```
* - {py:obj}`_unique_distributions <src.ui.pages.data_explorer_charts._unique_distributions>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._unique_distributions
    :summary:
    ```
* - {py:obj}`_render_visualization_tab <src.ui.pages.data_explorer_charts._render_visualization_tab>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_visualization_tab
    :summary:
    ```
* - {py:obj}`_render_line_bar_chart <src.ui.pages.data_explorer_charts._render_line_bar_chart>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_line_bar_chart
    :summary:
    ```
* - {py:obj}`_render_scatter_chart <src.ui.pages.data_explorer_charts._render_scatter_chart>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_scatter_chart
    :summary:
    ```
* - {py:obj}`_render_area_chart <src.ui.pages.data_explorer_charts._render_area_chart>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_area_chart
    :summary:
    ```
* - {py:obj}`_render_selected_chart <src.ui.pages.data_explorer_charts._render_selected_chart>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_selected_chart
    :summary:
    ```
* - {py:obj}`_find_td_table <src.ui.pages.data_explorer_charts._find_td_table>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._find_td_table
    :summary:
    ```
* - {py:obj}`_render_td_coord_section <src.ui.pages.data_explorer_charts._render_td_coord_section>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_td_coord_section
    :summary:
    ```
* - {py:obj}`_render_td_dist_section <src.ui.pages.data_explorer_charts._render_td_dist_section>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_td_dist_section
    :summary:
    ```
* - {py:obj}`_render_td_overview_tab <src.ui.pages.data_explorer_charts._render_td_overview_tab>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_td_overview_tab
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_CHART_TYPES <src.ui.pages.data_explorer_charts._CHART_TYPES>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer_charts._CHART_TYPES
    :summary:
    ```
````

### API

````{py:data} _CHART_TYPES
:canonical: src.ui.pages.data_explorer_charts._CHART_TYPES
:value: >
   ['Line Chart', 'Bar Chart', 'Scatter Plot', 'Area Chart', 'Histogram', 'Box Plot', 'Heatmap', 'Corre...

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._CHART_TYPES
```

````

````{py:function} _resolve_column(columns: typing.List[typing.Any], col_text: str) -> typing.Any
:canonical: src.ui.pages.data_explorer_charts._resolve_column

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._resolve_column
```
````

````{py:function} _numeric_columns(df: pandas.DataFrame) -> typing.List[str]
:canonical: src.ui.pages.data_explorer_charts._numeric_columns

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._numeric_columns
```
````

````{py:function} _has_distribution_meta(df: pandas.DataFrame) -> bool
:canonical: src.ui.pages.data_explorer_charts._has_distribution_meta

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._has_distribution_meta
```
````

````{py:function} _unique_distributions(df: pandas.DataFrame) -> typing.List[str]
:canonical: src.ui.pages.data_explorer_charts._unique_distributions

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._unique_distributions
```
````

````{py:function} _render_visualization_tab(df: pandas.DataFrame) -> None
:canonical: src.ui.pages.data_explorer_charts._render_visualization_tab

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_visualization_tab
```
````

````{py:function} _render_line_bar_chart(df: pandas.DataFrame, chart_type: str, x_col: str, y_cols: typing.List[str]) -> None
:canonical: src.ui.pages.data_explorer_charts._render_line_bar_chart

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_line_bar_chart
```
````

````{py:function} _render_scatter_chart(df: pandas.DataFrame, x_col: str, y_col: str, extra_opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.ui.pages.data_explorer_charts._render_scatter_chart

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_scatter_chart
```
````

````{py:function} _render_area_chart(df: pandas.DataFrame, x_col: str, y_col: str) -> None
:canonical: src.ui.pages.data_explorer_charts._render_area_chart

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_area_chart
```
````

````{py:function} _render_selected_chart(df: pandas.DataFrame, chart_type: str, x_col: str, extra_opts: typing.Dict[str, typing.Any], local_vars: typing.Dict[str, typing.Any]) -> None
:canonical: src.ui.pages.data_explorer_charts._render_selected_chart

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_selected_chart
```
````

````{py:function} _find_td_table(tables: typing.Dict[str, pandas.DataFrame], key: str) -> typing.Optional[pandas.DataFrame]
:canonical: src.ui.pages.data_explorer_charts._find_td_table

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._find_td_table
```
````

````{py:function} _render_td_coord_section(tables: typing.Dict[str, pandas.DataFrame], coord_keys: typing.List[str], depot_keys: typing.List[str], lazy_loader: typing.Optional[typing.Callable[[str], typing.Optional[pandas.DataFrame]]] = None) -> None
:canonical: src.ui.pages.data_explorer_charts._render_td_coord_section

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_td_coord_section
```
````

````{py:function} _render_td_dist_section(tables: typing.Dict[str, pandas.DataFrame], scalar_keys: typing.List[str], lazy_loader: typing.Optional[typing.Callable[[str], typing.Optional[pandas.DataFrame]]] = None) -> None
:canonical: src.ui.pages.data_explorer_charts._render_td_dist_section

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_td_dist_section
```
````

````{py:function} _render_td_overview_tab(td_meta: typing.Dict[str, typing.Any], tables: typing.Dict[str, pandas.DataFrame], lazy_loader: typing.Optional[typing.Callable[[str], typing.Optional[pandas.DataFrame]]] = None) -> None
:canonical: src.ui.pages.data_explorer_charts._render_td_overview_tab

```{autodoc2-docstring} src.ui.pages.data_explorer_charts._render_td_overview_tab
```
````
