# {py:mod}`src.ui.pages.simulation_summary`

```{py:module} src.ui.pages.simulation_summary
```

```{autodoc2-docstring} src.ui.pages.simulation_summary
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_discover_output_dirs <src.ui.pages.simulation_summary._discover_output_dirs>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._discover_output_dirs
    :summary:
    ```
* - {py:obj}`_load_json <src.ui.pages.simulation_summary._load_json>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._load_json
    :summary:
    ```
* - {py:obj}`_find_json_files <src.ui.pages.simulation_summary._find_json_files>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._find_json_files
    :summary:
    ```
* - {py:obj}`_parse_policy_name <src.ui.pages.simulation_summary._parse_policy_name>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._parse_policy_name
    :summary:
    ```
* - {py:obj}`_extract_distributions <src.ui.pages.simulation_summary._extract_distributions>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._extract_distributions
    :summary:
    ```
* - {py:obj}`_build_summary_df <src.ui.pages.simulation_summary._build_summary_df>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._build_summary_df
    :summary:
    ```
* - {py:obj}`_build_daily_df <src.ui.pages.simulation_summary._build_daily_df>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._build_daily_df
    :summary:
    ```
* - {py:obj}`_render_sidebar_controls <src.ui.pages.simulation_summary._render_sidebar_controls>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._render_sidebar_controls
    :summary:
    ```
* - {py:obj}`render_simulation_summary <src.ui.pages.simulation_summary.render_simulation_summary>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary.render_simulation_summary
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DIST_PATTERN <src.ui.pages.simulation_summary._DIST_PATTERN>`
  - ```{autodoc2-docstring} src.ui.pages.simulation_summary._DIST_PATTERN
    :summary:
    ```
````

### API

````{py:data} _DIST_PATTERN
:canonical: src.ui.pages.simulation_summary._DIST_PATTERN
:value: >
   'compile(...)'

```{autodoc2-docstring} src.ui.pages.simulation_summary._DIST_PATTERN
```

````

````{py:function} _discover_output_dirs() -> typing.List[str]
:canonical: src.ui.pages.simulation_summary._discover_output_dirs

```{autodoc2-docstring} src.ui.pages.simulation_summary._discover_output_dirs
```
````

````{py:function} _load_json(path: str) -> typing.Any
:canonical: src.ui.pages.simulation_summary._load_json

```{autodoc2-docstring} src.ui.pages.simulation_summary._load_json
```
````

````{py:function} _find_json_files(output_dir: str) -> typing.Dict[str, str]
:canonical: src.ui.pages.simulation_summary._find_json_files

```{autodoc2-docstring} src.ui.pages.simulation_summary._find_json_files
```
````

````{py:function} _parse_policy_name(raw_name: str) -> typing.Tuple[str, str]
:canonical: src.ui.pages.simulation_summary._parse_policy_name

```{autodoc2-docstring} src.ui.pages.simulation_summary._parse_policy_name
```
````

````{py:function} _extract_distributions(mean_data: typing.Dict[str, typing.Any]) -> typing.List[str]
:canonical: src.ui.pages.simulation_summary._extract_distributions

```{autodoc2-docstring} src.ui.pages.simulation_summary._extract_distributions
```
````

````{py:function} _build_summary_df(mean_data: typing.Dict[str, typing.Dict[str, float]], std_data: typing.Optional[typing.Dict[str, typing.Dict[str, float]]] = None) -> pandas.DataFrame
:canonical: src.ui.pages.simulation_summary._build_summary_df

```{autodoc2-docstring} src.ui.pages.simulation_summary._build_summary_df
```
````

````{py:function} _build_daily_df(daily_data: typing.Dict[str, typing.Dict[str, typing.Any]]) -> pandas.DataFrame
:canonical: src.ui.pages.simulation_summary._build_daily_df

```{autodoc2-docstring} src.ui.pages.simulation_summary._build_daily_df
```
````

````{py:function} _render_sidebar_controls(available_dirs: typing.List[str], distributions: typing.List[str]) -> typing.Dict[str, typing.Any]
:canonical: src.ui.pages.simulation_summary._render_sidebar_controls

```{autodoc2-docstring} src.ui.pages.simulation_summary._render_sidebar_controls
```
````

````{py:function} render_simulation_summary() -> None
:canonical: src.ui.pages.simulation_summary.render_simulation_summary

```{autodoc2-docstring} src.ui.pages.simulation_summary.render_simulation_summary
```
````
