# {py:mod}`src.pipeline.features.test.config`

```{py:module} src.pipeline.features.test.config
```

```{autodoc2-docstring} src.pipeline.features.test.config
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`expand_policy_configs <src.pipeline.features.test.config.expand_policy_configs>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config.expand_policy_configs
    :summary:
    ```
* - {py:obj}`_is_mapping <src.pipeline.features.test.config._is_mapping>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._is_mapping
    :summary:
    ```
* - {py:obj}`_is_sequence <src.pipeline.features.test.config._is_sequence>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._is_sequence
    :summary:
    ```
* - {py:obj}`_selection_keys_set_by <src.pipeline.features.test.config._selection_keys_set_by>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._selection_keys_set_by
    :summary:
    ```
* - {py:obj}`_first_keyed_value <src.pipeline.features.test.config._first_keyed_value>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._first_keyed_value
    :summary:
    ```
* - {py:obj}`_overwrite_key <src.pipeline.features.test.config._overwrite_key>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._overwrite_key
    :summary:
    ```
* - {py:obj}`_apply_caller_selection <src.pipeline.features.test.config._apply_caller_selection>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._apply_caller_selection
    :summary:
    ```
* - {py:obj}`_collapse_selection_variants <src.pipeline.features.test.config._collapse_selection_variants>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._collapse_selection_variants
    :summary:
    ```
* - {py:obj}`_pin_variant_selection <src.pipeline.features.test.config._pin_variant_selection>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._pin_variant_selection
    :summary:
    ```
* - {py:obj}`_resolve_policy_cfg_path <src.pipeline.features.test.config._resolve_policy_cfg_path>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._resolve_policy_cfg_path
    :summary:
    ```
* - {py:obj}`_extract_variants <src.pipeline.features.test.config._extract_variants>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._extract_variants
    :summary:
    ```
* - {py:obj}`_find_inner_config <src.pipeline.features.test.config._find_inner_config>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._find_inner_config
    :summary:
    ```
* - {py:obj}`_parse_inner_components <src.pipeline.features.test.config._parse_inner_components>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._parse_inner_components
    :summary:
    ```
* - {py:obj}`_apply_overrides <src.pipeline.features.test.config._apply_overrides>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._apply_overrides
    :summary:
    ```
* - {py:obj}`_expand_dict_ms_list <src.pipeline.features.test.config._expand_dict_ms_list>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._expand_dict_ms_list
    :summary:
    ```
* - {py:obj}`_clean_id <src.pipeline.features.test.config._clean_id>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._clean_id
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SELECTION_KEYS <src.pipeline.features.test.config._SELECTION_KEYS>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._SELECTION_KEYS
    :summary:
    ```
````

### API

````{py:function} expand_policy_configs(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.test.config.expand_policy_configs

```{autodoc2-docstring} src.pipeline.features.test.config.expand_policy_configs
```
````

````{py:data} _SELECTION_KEYS
:canonical: src.pipeline.features.test.config._SELECTION_KEYS
:value: >
   ('mandatory_selection', 'acceptance_criteria')

```{autodoc2-docstring} src.pipeline.features.test.config._SELECTION_KEYS
```

````

````{py:function} _is_mapping(node: typing.Any) -> bool
:canonical: src.pipeline.features.test.config._is_mapping

```{autodoc2-docstring} src.pipeline.features.test.config._is_mapping
```
````

````{py:function} _is_sequence(node: typing.Any) -> bool
:canonical: src.pipeline.features.test.config._is_sequence

```{autodoc2-docstring} src.pipeline.features.test.config._is_sequence
```
````

````{py:function} _selection_keys_set_by(node: typing.Any) -> typing.Set[str]
:canonical: src.pipeline.features.test.config._selection_keys_set_by

```{autodoc2-docstring} src.pipeline.features.test.config._selection_keys_set_by
```
````

````{py:function} _first_keyed_value(node: typing.Any, key: str) -> typing.Any
:canonical: src.pipeline.features.test.config._first_keyed_value

```{autodoc2-docstring} src.pipeline.features.test.config._first_keyed_value
```
````

````{py:function} _overwrite_key(node: typing.Any, key: str, value: typing.Any) -> None
:canonical: src.pipeline.features.test.config._overwrite_key

```{autodoc2-docstring} src.pipeline.features.test.config._overwrite_key
```
````

````{py:function} _apply_caller_selection(final_cfg: typing.Any, overrides: typing.Any) -> None
:canonical: src.pipeline.features.test.config._apply_caller_selection

```{autodoc2-docstring} src.pipeline.features.test.config._apply_caller_selection
```
````

````{py:function} _collapse_selection_variants(variants: typing.List[typing.Tuple[str, str, typing.Any]]) -> typing.List[typing.Tuple[str, str, typing.Any]]
:canonical: src.pipeline.features.test.config._collapse_selection_variants

```{autodoc2-docstring} src.pipeline.features.test.config._collapse_selection_variants
```
````

````{py:function} _pin_variant_selection(obj: typing.Any, var_cfg: typing.Any, protected: typing.Optional[typing.Set[str]] = None) -> None
:canonical: src.pipeline.features.test.config._pin_variant_selection

```{autodoc2-docstring} src.pipeline.features.test.config._pin_variant_selection
```
````

````{py:function} _resolve_policy_cfg_path(pol_name: str) -> str
:canonical: src.pipeline.features.test.config._resolve_policy_cfg_path

```{autodoc2-docstring} src.pipeline.features.test.config._resolve_policy_cfg_path
```
````

````{py:function} _extract_variants(pol_name: str, cfg_path: str) -> typing.Tuple[typing.List[typing.Tuple[str, str, typing.Any]], typing.Any]
:canonical: src.pipeline.features.test.config._extract_variants

```{autodoc2-docstring} src.pipeline.features.test.config._extract_variants
```
````

````{py:function} _find_inner_config(pol_cfg: typing.Any, pol_name: str = '') -> typing.Tuple[typing.Any, typing.Any]
:canonical: src.pipeline.features.test.config._find_inner_config

```{autodoc2-docstring} src.pipeline.features.test.config._find_inner_config
```
````

````{py:function} _parse_inner_components(inner_cfg: typing.Any) -> typing.Tuple[typing.List[typing.Any], typing.List[typing.Any], typing.List[typing.Any], int, int]
:canonical: src.pipeline.features.test.config._parse_inner_components

```{autodoc2-docstring} src.pipeline.features.test.config._parse_inner_components
```
````

````{py:function} _apply_overrides(var_cfg: typing.Any, ms_idx: int, ms_item: typing.Any, ac_idx: int, ac_item: typing.Any) -> None
:canonical: src.pipeline.features.test.config._apply_overrides

```{autodoc2-docstring} src.pipeline.features.test.config._apply_overrides
```
````

````{py:function} _expand_dict_ms_list(ms_list: typing.Any) -> typing.List[typing.Any]
:canonical: src.pipeline.features.test.config._expand_dict_ms_list

```{autodoc2-docstring} src.pipeline.features.test.config._expand_dict_ms_list
```
````

````{py:function} _clean_id(path_or_str: typing.Any, prefix: str) -> str
:canonical: src.pipeline.features.test.config._clean_id

```{autodoc2-docstring} src.pipeline.features.test.config._clean_id
```
````
