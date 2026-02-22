# {py:mod}`src.tracking.logging.excel_summary`

```{py:module} src.tracking.logging.excel_summary
```

```{autodoc2-docstring} src.tracking.logging.excel_summary
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_parse_policy_name <src.tracking.logging.excel_summary._parse_policy_name>`
  - ```{autodoc2-docstring} src.tracking.logging.excel_summary._parse_policy_name
    :summary:
    ```
* - {py:obj}`_load_json <src.tracking.logging.excel_summary._load_json>`
  - ```{autodoc2-docstring} src.tracking.logging.excel_summary._load_json
    :summary:
    ```
* - {py:obj}`discover_and_aggregate <src.tracking.logging.excel_summary.discover_and_aggregate>`
  - ```{autodoc2-docstring} src.tracking.logging.excel_summary.discover_and_aggregate
    :summary:
    ```
* - {py:obj}`main <src.tracking.logging.excel_summary.main>`
  - ```{autodoc2-docstring} src.tracking.logging.excel_summary.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DIST_PATTERN <src.tracking.logging.excel_summary._DIST_PATTERN>`
  - ```{autodoc2-docstring} src.tracking.logging.excel_summary._DIST_PATTERN
    :summary:
    ```
* - {py:obj}`_DISPLAY_METRICS <src.tracking.logging.excel_summary._DISPLAY_METRICS>`
  - ```{autodoc2-docstring} src.tracking.logging.excel_summary._DISPLAY_METRICS
    :summary:
    ```
````

### API

````{py:data} _DIST_PATTERN
:canonical: src.tracking.logging.excel_summary._DIST_PATTERN
:value: >
   'compile(...)'

```{autodoc2-docstring} src.tracking.logging.excel_summary._DIST_PATTERN
```

````

````{py:data} _DISPLAY_METRICS
:canonical: src.tracking.logging.excel_summary._DISPLAY_METRICS
:value: >
   ['profit', 'cost', 'kg', 'km', 'kg/km', 'overflows', 'ncol', 'kg_lost', 'days', 'time']

```{autodoc2-docstring} src.tracking.logging.excel_summary._DISPLAY_METRICS
```

````

````{py:function} _parse_policy_name(raw_name: str) -> typing.Tuple[str, str]
:canonical: src.tracking.logging.excel_summary._parse_policy_name

```{autodoc2-docstring} src.tracking.logging.excel_summary._parse_policy_name
```
````

````{py:function} _load_json(path: str) -> typing.Any
:canonical: src.tracking.logging.excel_summary._load_json

```{autodoc2-docstring} src.tracking.logging.excel_summary._load_json
```
````

````{py:function} discover_and_aggregate() -> pandas.DataFrame
:canonical: src.tracking.logging.excel_summary.discover_and_aggregate

```{autodoc2-docstring} src.tracking.logging.excel_summary.discover_and_aggregate
```
````

````{py:function} main() -> None
:canonical: src.tracking.logging.excel_summary.main

```{autodoc2-docstring} src.tracking.logging.excel_summary.main
```
````
