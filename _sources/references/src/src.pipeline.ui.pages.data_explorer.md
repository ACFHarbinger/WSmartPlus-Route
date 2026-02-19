# {py:mod}`src.pipeline.ui.pages.data_explorer`

```{py:module} src.pipeline.ui.pages.data_explorer
```

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_process_raw_to_dfs <src.pipeline.ui.pages.data_explorer._process_raw_to_dfs>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._process_raw_to_dfs
    :summary:
    ```
* - {py:obj}`_try_vrpp_split <src.pipeline.ui.pages.data_explorer._try_vrpp_split>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._try_vrpp_split
    :summary:
    ```
* - {py:obj}`_pivot_json_data <src.pipeline.ui.pages.data_explorer._pivot_json_data>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._pivot_json_data
    :summary:
    ```
* - {py:obj}`_load_json_file <src.pipeline.ui.pages.data_explorer._load_json_file>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_json_file
    :summary:
    ```
* - {py:obj}`_load_jsonl_file <src.pipeline.ui.pages.data_explorer._load_jsonl_file>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_jsonl_file
    :summary:
    ```
* - {py:obj}`_load_npz_file <src.pipeline.ui.pages.data_explorer._load_npz_file>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_npz_file
    :summary:
    ```
* - {py:obj}`_load_uploaded_file <src.pipeline.ui.pages.data_explorer._load_uploaded_file>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_uploaded_file
    :summary:
    ```
* - {py:obj}`_resolve_column <src.pipeline.ui.pages.data_explorer._resolve_column>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._resolve_column
    :summary:
    ```
* - {py:obj}`_numeric_columns <src.pipeline.ui.pages.data_explorer._numeric_columns>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._numeric_columns
    :summary:
    ```
* - {py:obj}`_safe_nunique <src.pipeline.ui.pages.data_explorer._safe_nunique>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._safe_nunique
    :summary:
    ```
* - {py:obj}`_has_distribution_meta <src.pipeline.ui.pages.data_explorer._has_distribution_meta>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._has_distribution_meta
    :summary:
    ```
* - {py:obj}`_unique_distributions <src.pipeline.ui.pages.data_explorer._unique_distributions>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._unique_distributions
    :summary:
    ```
* - {py:obj}`_render_raw_data_tab <src.pipeline.ui.pages.data_explorer._render_raw_data_tab>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_raw_data_tab
    :summary:
    ```
* - {py:obj}`_render_statistics_tab <src.pipeline.ui.pages.data_explorer._render_statistics_tab>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_statistics_tab
    :summary:
    ```
* - {py:obj}`_render_correlation_tab <src.pipeline.ui.pages.data_explorer._render_correlation_tab>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_correlation_tab
    :summary:
    ```
* - {py:obj}`_render_visualization_tab <src.pipeline.ui.pages.data_explorer._render_visualization_tab>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_visualization_tab
    :summary:
    ```
* - {py:obj}`_render_line_bar_chart <src.pipeline.ui.pages.data_explorer._render_line_bar_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_line_bar_chart
    :summary:
    ```
* - {py:obj}`_render_scatter_chart <src.pipeline.ui.pages.data_explorer._render_scatter_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_scatter_chart
    :summary:
    ```
* - {py:obj}`_render_area_chart <src.pipeline.ui.pages.data_explorer._render_area_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_area_chart
    :summary:
    ```
* - {py:obj}`_render_selected_chart <src.pipeline.ui.pages.data_explorer._render_selected_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_selected_chart
    :summary:
    ```
* - {py:obj}`_render_sidebar_controls <src.pipeline.ui.pages.data_explorer._render_sidebar_controls>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_sidebar_controls
    :summary:
    ```
* - {py:obj}`render_data_explorer <src.pipeline.ui.pages.data_explorer.render_data_explorer>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer.render_data_explorer
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_CHART_TYPES <src.pipeline.ui.pages.data_explorer._CHART_TYPES>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._CHART_TYPES
    :summary:
    ```
* - {py:obj}`_META_COLUMNS <src.pipeline.ui.pages.data_explorer._META_COLUMNS>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._META_COLUMNS
    :summary:
    ```
* - {py:obj}`_DIST_PATTERN <src.pipeline.ui.pages.data_explorer._DIST_PATTERN>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._DIST_PATTERN
    :summary:
    ```
````

### API

````{py:data} _CHART_TYPES
:canonical: src.pipeline.ui.pages.data_explorer._CHART_TYPES
:value: >
   ['Line Chart', 'Bar Chart', 'Scatter Plot', 'Area Chart', 'Histogram', 'Box Plot', 'Heatmap', 'Corre...

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._CHART_TYPES
```

````

````{py:data} _META_COLUMNS
:canonical: src.pipeline.ui.pages.data_explorer._META_COLUMNS
:value: >
   ('__Policy_Names__', '__Distributions__', '__File_IDs__')

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._META_COLUMNS
```

````

````{py:function} _process_raw_to_dfs(raw_data: typing.Any) -> typing.List[pandas.DataFrame]
:canonical: src.pipeline.ui.pages.data_explorer._process_raw_to_dfs

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._process_raw_to_dfs
```
````

````{py:function} _try_vrpp_split(df: pandas.DataFrame) -> typing.Optional[typing.List[typing.Tuple[str, pandas.DataFrame]]]
:canonical: src.pipeline.ui.pages.data_explorer._try_vrpp_split

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._try_vrpp_split
```
````

````{py:data} _DIST_PATTERN
:canonical: src.pipeline.ui.pages.data_explorer._DIST_PATTERN
:value: >
   'compile(...)'

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._DIST_PATTERN
```

````

````{py:function} _pivot_json_data(data: typing.Dict[str, typing.Any], file_id: str = '') -> typing.Dict[str, typing.List[typing.Any]]
:canonical: src.pipeline.ui.pages.data_explorer._pivot_json_data

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._pivot_json_data
```
````

````{py:function} _load_json_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.pipeline.ui.pages.data_explorer._load_json_file

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_json_file
```
````

````{py:function} _load_jsonl_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.pipeline.ui.pages.data_explorer._load_jsonl_file

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_jsonl_file
```
````

````{py:function} _load_npz_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.pipeline.ui.pages.data_explorer._load_npz_file

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_npz_file
```
````

````{py:function} _load_uploaded_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.pipeline.ui.pages.data_explorer._load_uploaded_file

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_uploaded_file
```
````

````{py:function} _resolve_column(columns: typing.List[typing.Any], col_text: str) -> typing.Any
:canonical: src.pipeline.ui.pages.data_explorer._resolve_column

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._resolve_column
```
````

````{py:function} _numeric_columns(df: pandas.DataFrame) -> typing.List[str]
:canonical: src.pipeline.ui.pages.data_explorer._numeric_columns

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._numeric_columns
```
````

````{py:function} _safe_nunique(s: pandas.Series) -> int
:canonical: src.pipeline.ui.pages.data_explorer._safe_nunique

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._safe_nunique
```
````

````{py:function} _has_distribution_meta(df: pandas.DataFrame) -> bool
:canonical: src.pipeline.ui.pages.data_explorer._has_distribution_meta

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._has_distribution_meta
```
````

````{py:function} _unique_distributions(df: pandas.DataFrame) -> typing.List[str]
:canonical: src.pipeline.ui.pages.data_explorer._unique_distributions

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._unique_distributions
```
````

````{py:function} _render_raw_data_tab(df: pandas.DataFrame, selected_table: str, visible_columns: typing.Optional[typing.List[str]], row_limit: int, precision: int) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_raw_data_tab

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_raw_data_tab
```
````

````{py:function} _render_statistics_tab(df: pandas.DataFrame) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_statistics_tab

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_statistics_tab
```
````

````{py:function} _render_correlation_tab(df: pandas.DataFrame) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_correlation_tab

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_correlation_tab
```
````

````{py:function} _render_visualization_tab(df: pandas.DataFrame) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_visualization_tab

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_visualization_tab
```
````

````{py:function} _render_line_bar_chart(df: pandas.DataFrame, chart_type: str, x_col: str, y_cols: typing.List[str]) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_line_bar_chart

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_line_bar_chart
```
````

````{py:function} _render_scatter_chart(df: pandas.DataFrame, x_col: str, y_col: str, extra_opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_scatter_chart

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_scatter_chart
```
````

````{py:function} _render_area_chart(df: pandas.DataFrame, x_col: str, y_col: str) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_area_chart

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_area_chart
```
````

````{py:function} _render_selected_chart(df: pandas.DataFrame, chart_type: str, x_col: str, extra_opts: typing.Dict[str, typing.Any], local_vars: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_selected_chart

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_selected_chart
```
````

````{py:function} _render_sidebar_controls(df: pandas.DataFrame) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.ui.pages.data_explorer._render_sidebar_controls

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_sidebar_controls
```
````

````{py:function} render_data_explorer() -> None
:canonical: src.pipeline.ui.pages.data_explorer.render_data_explorer

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer.render_data_explorer
```
````
