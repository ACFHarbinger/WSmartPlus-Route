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
* - {py:obj}`_clean_id <src.pipeline.features.test.config._clean_id>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._clean_id
    :summary:
    ```
````

### API

````{py:function} expand_policy_configs(cfg: logic.src.configs.Config) -> None
:canonical: src.pipeline.features.test.config.expand_policy_configs

```{autodoc2-docstring} src.pipeline.features.test.config.expand_policy_configs
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

````{py:function} _clean_id(path_or_str: typing.Any, prefix: str) -> str
:canonical: src.pipeline.features.test.config._clean_id

```{autodoc2-docstring} src.pipeline.features.test.config._clean_id
```
````
