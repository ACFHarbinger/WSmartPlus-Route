# {py:mod}`src.ui.pages.data_explorer`

```{py:module} src.ui.pages.data_explorer
```

```{autodoc2-docstring} src.ui.pages.data_explorer
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_td_tensor_to_df <src.ui.pages.data_explorer._td_tensor_to_df>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._td_tensor_to_df
    :summary:
    ```
* - {py:obj}`_collect_td_metadata <src.ui.pages.data_explorer._collect_td_metadata>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._collect_td_metadata
    :summary:
    ```
* - {py:obj}`_load_td_file <src.ui.pages.data_explorer._load_td_file>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._load_td_file
    :summary:
    ```
* - {py:obj}`_process_raw_to_dfs <src.ui.pages.data_explorer._process_raw_to_dfs>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._process_raw_to_dfs
    :summary:
    ```
* - {py:obj}`_try_vrpp_split <src.ui.pages.data_explorer._try_vrpp_split>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._try_vrpp_split
    :summary:
    ```
* - {py:obj}`_pivot_json_data <src.ui.pages.data_explorer._pivot_json_data>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._pivot_json_data
    :summary:
    ```
* - {py:obj}`_load_json_file <src.ui.pages.data_explorer._load_json_file>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._load_json_file
    :summary:
    ```
* - {py:obj}`_load_jsonl_file <src.ui.pages.data_explorer._load_jsonl_file>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._load_jsonl_file
    :summary:
    ```
* - {py:obj}`_load_npz_file <src.ui.pages.data_explorer._load_npz_file>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._load_npz_file
    :summary:
    ```
* - {py:obj}`_load_uploaded_file <src.ui.pages.data_explorer._load_uploaded_file>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._load_uploaded_file
    :summary:
    ```
* - {py:obj}`_load_td_from_path <src.ui.pages.data_explorer._load_td_from_path>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._load_td_from_path
    :summary:
    ```
* - {py:obj}`_lazy_load_td_tensor <src.ui.pages.data_explorer._lazy_load_td_tensor>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._lazy_load_td_tensor
    :summary:
    ```
* - {py:obj}`_safe_nunique <src.ui.pages.data_explorer._safe_nunique>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._safe_nunique
    :summary:
    ```
* - {py:obj}`_render_raw_data_tab <src.ui.pages.data_explorer._render_raw_data_tab>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._render_raw_data_tab
    :summary:
    ```
* - {py:obj}`_render_statistics_tab <src.ui.pages.data_explorer._render_statistics_tab>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._render_statistics_tab
    :summary:
    ```
* - {py:obj}`_render_correlation_tab <src.ui.pages.data_explorer._render_correlation_tab>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._render_correlation_tab
    :summary:
    ```
* - {py:obj}`_render_sidebar_controls <src.ui.pages.data_explorer._render_sidebar_controls>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._render_sidebar_controls
    :summary:
    ```
* - {py:obj}`_resolve_selected_df <src.ui.pages.data_explorer._resolve_selected_df>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._resolve_selected_df
    :summary:
    ```
* - {py:obj}`render_data_explorer <src.ui.pages.data_explorer.render_data_explorer>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer.render_data_explorer
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_META_COLUMNS <src.ui.pages.data_explorer._META_COLUMNS>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._META_COLUMNS
    :summary:
    ```
* - {py:obj}`_MAX_WIDE_COLS <src.ui.pages.data_explorer._MAX_WIDE_COLS>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._MAX_WIDE_COLS
    :summary:
    ```
* - {py:obj}`_DIST_PATTERN <src.ui.pages.data_explorer._DIST_PATTERN>`
  - ```{autodoc2-docstring} src.ui.pages.data_explorer._DIST_PATTERN
    :summary:
    ```
````

### API

````{py:data} _META_COLUMNS
:canonical: src.ui.pages.data_explorer._META_COLUMNS
:value: >
   ('__Policy_Names__', '__Distributions__', '__File_IDs__')

```{autodoc2-docstring} src.ui.pages.data_explorer._META_COLUMNS
```

````

````{py:data} _MAX_WIDE_COLS
:canonical: src.ui.pages.data_explorer._MAX_WIDE_COLS
:value: >
   200

```{autodoc2-docstring} src.ui.pages.data_explorer._MAX_WIDE_COLS
```

````

````{py:function} _td_tensor_to_df(key: str, arr: numpy.ndarray) -> typing.Optional[pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._td_tensor_to_df

```{autodoc2-docstring} src.ui.pages.data_explorer._td_tensor_to_df
```
````

````{py:function} _collect_td_metadata(td: typing.Any, keys: typing.List[str]) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[str], typing.List[typing.Dict[str, typing.Any]]]
:canonical: src.ui.pages.data_explorer._collect_td_metadata

```{autodoc2-docstring} src.ui.pages.data_explorer._collect_td_metadata
```
````

````{py:function} _load_td_file(uploaded_file: typing.Any, cache_key: str) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._load_td_file

```{autodoc2-docstring} src.ui.pages.data_explorer._load_td_file
```
````

````{py:function} _process_raw_to_dfs(raw_data: typing.Any) -> typing.List[pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._process_raw_to_dfs

```{autodoc2-docstring} src.ui.pages.data_explorer._process_raw_to_dfs
```
````

````{py:function} _try_vrpp_split(df: pandas.DataFrame) -> typing.Optional[typing.List[typing.Tuple[str, pandas.DataFrame]]]
:canonical: src.ui.pages.data_explorer._try_vrpp_split

```{autodoc2-docstring} src.ui.pages.data_explorer._try_vrpp_split
```
````

````{py:data} _DIST_PATTERN
:canonical: src.ui.pages.data_explorer._DIST_PATTERN
:value: >
   'compile(...)'

```{autodoc2-docstring} src.ui.pages.data_explorer._DIST_PATTERN
```

````

````{py:function} _pivot_json_data(data: typing.Dict[str, typing.Any], file_id: str = '') -> typing.Dict[str, typing.List[typing.Any]]
:canonical: src.ui.pages.data_explorer._pivot_json_data

```{autodoc2-docstring} src.ui.pages.data_explorer._pivot_json_data
```
````

````{py:function} _load_json_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._load_json_file

```{autodoc2-docstring} src.ui.pages.data_explorer._load_json_file
```
````

````{py:function} _load_jsonl_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._load_jsonl_file

```{autodoc2-docstring} src.ui.pages.data_explorer._load_jsonl_file
```
````

````{py:function} _load_npz_file(uploaded_file: typing.Any) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._load_npz_file

```{autodoc2-docstring} src.ui.pages.data_explorer._load_npz_file
```
````

````{py:function} _load_uploaded_file(uploaded_file: typing.Any, cache_key: str = '') -> typing.Dict[str, pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._load_uploaded_file

```{autodoc2-docstring} src.ui.pages.data_explorer._load_uploaded_file
```
````

````{py:function} _load_td_from_path(path: str, cache_key: str) -> typing.Dict[str, pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._load_td_from_path

```{autodoc2-docstring} src.ui.pages.data_explorer._load_td_from_path
```
````

````{py:function} _lazy_load_td_tensor(td_meta: typing.Dict[str, typing.Any], key: str, cache_key: str) -> typing.Optional[pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._lazy_load_td_tensor

```{autodoc2-docstring} src.ui.pages.data_explorer._lazy_load_td_tensor
```
````

````{py:function} _safe_nunique(s: pandas.Series) -> int
:canonical: src.ui.pages.data_explorer._safe_nunique

```{autodoc2-docstring} src.ui.pages.data_explorer._safe_nunique
```
````

````{py:function} _render_raw_data_tab(df: pandas.DataFrame, selected_table: str, visible_columns: typing.Optional[typing.List[str]], row_limit: int, precision: int) -> None
:canonical: src.ui.pages.data_explorer._render_raw_data_tab

```{autodoc2-docstring} src.ui.pages.data_explorer._render_raw_data_tab
```
````

````{py:function} _render_statistics_tab(df: pandas.DataFrame) -> None
:canonical: src.ui.pages.data_explorer._render_statistics_tab

```{autodoc2-docstring} src.ui.pages.data_explorer._render_statistics_tab
```
````

````{py:function} _render_correlation_tab(df: pandas.DataFrame) -> None
:canonical: src.ui.pages.data_explorer._render_correlation_tab

```{autodoc2-docstring} src.ui.pages.data_explorer._render_correlation_tab
```
````

````{py:function} _render_sidebar_controls(df: pandas.DataFrame) -> typing.Dict[str, typing.Any]
:canonical: src.ui.pages.data_explorer._render_sidebar_controls

```{autodoc2-docstring} src.ui.pages.data_explorer._render_sidebar_controls
```
````

````{py:function} _resolve_selected_df(tables: typing.Dict[str, pandas.DataFrame], td_meta: typing.Dict[str, typing.Any], selected_table: str, cache_key: str) -> typing.Optional[pandas.DataFrame]
:canonical: src.ui.pages.data_explorer._resolve_selected_df

```{autodoc2-docstring} src.ui.pages.data_explorer._resolve_selected_df
```
````

````{py:function} render_data_explorer() -> None
:canonical: src.ui.pages.data_explorer.render_data_explorer

```{autodoc2-docstring} src.ui.pages.data_explorer.render_data_explorer
```
````
