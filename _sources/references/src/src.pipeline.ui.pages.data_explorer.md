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
* - {py:obj}`_load_uploaded_file <src.pipeline.ui.pages.data_explorer._load_uploaded_file>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_uploaded_file
    :summary:
    ```
* - {py:obj}`_render_chart <src.pipeline.ui.pages.data_explorer._render_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_chart
    :summary:
    ```
* - {py:obj}`_resolve_column <src.pipeline.ui.pages.data_explorer._resolve_column>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._resolve_column
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
````

### API

````{py:data} _CHART_TYPES
:canonical: src.pipeline.ui.pages.data_explorer._CHART_TYPES
:value: >
   ['Line Chart', 'Bar Chart', 'Scatter Plot', 'Area Chart', 'Heatmap']

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._CHART_TYPES
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

````{py:function} _load_uploaded_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.pipeline.ui.pages.data_explorer._load_uploaded_file

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._load_uploaded_file
```
````

````{py:function} _render_chart(df: pandas.DataFrame, chart_type: str, x_col: str, y_col: str) -> None
:canonical: src.pipeline.ui.pages.data_explorer._render_chart

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._render_chart
```
````

````{py:function} _resolve_column(columns: typing.List[typing.Any], col_text: str) -> typing.Any
:canonical: src.pipeline.ui.pages.data_explorer._resolve_column

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer._resolve_column
```
````

````{py:function} render_data_explorer() -> None
:canonical: src.pipeline.ui.pages.data_explorer.render_data_explorer

```{autodoc2-docstring} src.pipeline.ui.pages.data_explorer.render_data_explorer
```
````
