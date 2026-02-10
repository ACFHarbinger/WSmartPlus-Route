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
* - {py:obj}`_apply_mg_override <src.pipeline.features.test.config._apply_mg_override>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._apply_mg_override
    :summary:
    ```
* - {py:obj}`_clean_id <src.pipeline.features.test.config._clean_id>`
  - ```{autodoc2-docstring} src.pipeline.features.test.config._clean_id
    :summary:
    ```
````

### API

````{py:function} expand_policy_configs(opts)
:canonical: src.pipeline.features.test.config.expand_policy_configs

```{autodoc2-docstring} src.pipeline.features.test.config.expand_policy_configs
```
````

````{py:function} _resolve_policy_cfg_path(pol_name: str) -> str
:canonical: src.pipeline.features.test.config._resolve_policy_cfg_path

```{autodoc2-docstring} src.pipeline.features.test.config._resolve_policy_cfg_path
```
````

````{py:function} _extract_variants(pol_name: str, cfg_path: str)
:canonical: src.pipeline.features.test.config._extract_variants

```{autodoc2-docstring} src.pipeline.features.test.config._extract_variants
```
````

````{py:function} _find_inner_config(pol_cfg: typing.Any)
:canonical: src.pipeline.features.test.config._find_inner_config

```{autodoc2-docstring} src.pipeline.features.test.config._find_inner_config
```
````

````{py:function} _parse_inner_components(inner_cfg)
:canonical: src.pipeline.features.test.config._parse_inner_components

```{autodoc2-docstring} src.pipeline.features.test.config._parse_inner_components
```
````

````{py:function} _apply_mg_override(var_cfg: typing.Any, match_idx: int, mg_item: str)
:canonical: src.pipeline.features.test.config._apply_mg_override

```{autodoc2-docstring} src.pipeline.features.test.config._apply_mg_override
```
````

````{py:function} _clean_id(path_or_str: typing.Any, prefix: str) -> str
:canonical: src.pipeline.features.test.config._clean_id

```{autodoc2-docstring} src.pipeline.features.test.config._clean_id
```
````
