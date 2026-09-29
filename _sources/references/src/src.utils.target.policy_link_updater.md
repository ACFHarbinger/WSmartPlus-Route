# {py:mod}`src.utils.target.policy_link_updater`

```{py:module} src.utils.target.policy_link_updater
```

```{autodoc2-docstring} src.utils.target.policy_link_updater
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_resolve_stem <src.utils.target.policy_link_updater._resolve_stem>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater._resolve_stem
    :summary:
    ```
* - {py:obj}`_field_re <src.utils.target.policy_link_updater._field_re>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater._field_re
    :summary:
    ```
* - {py:obj}`_list_available <src.utils.target.policy_link_updater._list_available>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater._list_available
    :summary:
    ```
* - {py:obj}`_list_keys <src.utils.target.policy_link_updater._list_keys>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater._list_keys
    :summary:
    ```
* - {py:obj}`_update <src.utils.target.policy_link_updater._update>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater._update
    :summary:
    ```
* - {py:obj}`list_available_ms_strategies <src.utils.target.policy_link_updater.list_available_ms_strategies>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater.list_available_ms_strategies
    :summary:
    ```
* - {py:obj}`list_strategy_keys <src.utils.target.policy_link_updater.list_strategy_keys>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater.list_strategy_keys
    :summary:
    ```
* - {py:obj}`update_mandatory_selection <src.utils.target.policy_link_updater.update_mandatory_selection>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater.update_mandatory_selection
    :summary:
    ```
* - {py:obj}`list_available_ri_improvers <src.utils.target.policy_link_updater.list_available_ri_improvers>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater.list_available_ri_improvers
    :summary:
    ```
* - {py:obj}`list_improver_keys <src.utils.target.policy_link_updater.list_improver_keys>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater.list_improver_keys
    :summary:
    ```
* - {py:obj}`update_route_improvement <src.utils.target.policy_link_updater.update_route_improvement>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater.update_route_improvement
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SCRIPT_DIR <src.utils.target.policy_link_updater._SCRIPT_DIR>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater._SCRIPT_DIR
    :summary:
    ```
* - {py:obj}`_DEFAULT_CONFIGS_DIR <src.utils.target.policy_link_updater._DEFAULT_CONFIGS_DIR>`
  - ```{autodoc2-docstring} src.utils.target.policy_link_updater._DEFAULT_CONFIGS_DIR
    :summary:
    ```
````

### API

````{py:data} _SCRIPT_DIR
:canonical: src.utils.target.policy_link_updater._SCRIPT_DIR
:value: >
   'dirname(...)'

```{autodoc2-docstring} src.utils.target.policy_link_updater._SCRIPT_DIR
```

````

````{py:data} _DEFAULT_CONFIGS_DIR
:canonical: src.utils.target.policy_link_updater._DEFAULT_CONFIGS_DIR
:value: >
   'normpath(...)'

```{autodoc2-docstring} src.utils.target.policy_link_updater._DEFAULT_CONFIGS_DIR
```

````

````{py:function} _resolve_stem(yaml_arg: str) -> str
:canonical: src.utils.target.policy_link_updater._resolve_stem

```{autodoc2-docstring} src.utils.target.policy_link_updater._resolve_stem
```
````

````{py:function} _field_re(field_label: str) -> re.Pattern
:canonical: src.utils.target.policy_link_updater._field_re

```{autodoc2-docstring} src.utils.target.policy_link_updater._field_re
```
````

````{py:function} _list_available(configs_dir: str, file_prefix: str) -> typing.List[str]
:canonical: src.utils.target.policy_link_updater._list_available

```{autodoc2-docstring} src.utils.target.policy_link_updater._list_available
```
````

````{py:function} _list_keys(yaml_stem: str, configs_dir: str) -> typing.List[str]
:canonical: src.utils.target.policy_link_updater._list_keys

```{autodoc2-docstring} src.utils.target.policy_link_updater._list_keys
```
````

````{py:function} _update(field_label: str, file_prefix: str, constructors: typing.List[str], yaml_stem_arg: str, keys: typing.List[str], configs_dir: str, dry_run: bool, verbose: bool, missing_file_noun: str) -> typing.List[typing.Tuple[str, int]]
:canonical: src.utils.target.policy_link_updater._update

```{autodoc2-docstring} src.utils.target.policy_link_updater._update
```
````

````{py:function} list_available_ms_strategies(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.policy_link_updater.list_available_ms_strategies

```{autodoc2-docstring} src.utils.target.policy_link_updater.list_available_ms_strategies
```
````

````{py:function} list_strategy_keys(ms_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.policy_link_updater.list_strategy_keys

```{autodoc2-docstring} src.utils.target.policy_link_updater.list_strategy_keys
```
````

````{py:function} update_mandatory_selection(constructors: typing.List[str], ms_yaml: str, keys: typing.List[str], configs_dir: str = _DEFAULT_CONFIGS_DIR, dry_run: bool = False, verbose: bool = True) -> typing.List[typing.Tuple[str, int]]
:canonical: src.utils.target.policy_link_updater.update_mandatory_selection

```{autodoc2-docstring} src.utils.target.policy_link_updater.update_mandatory_selection
```
````

````{py:function} list_available_ri_improvers(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.policy_link_updater.list_available_ri_improvers

```{autodoc2-docstring} src.utils.target.policy_link_updater.list_available_ri_improvers
```
````

````{py:function} list_improver_keys(ri_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.policy_link_updater.list_improver_keys

```{autodoc2-docstring} src.utils.target.policy_link_updater.list_improver_keys
```
````

````{py:function} update_route_improvement(constructors: typing.List[str], ri_yaml: str, keys: typing.List[str], configs_dir: str = _DEFAULT_CONFIGS_DIR, dry_run: bool = False, verbose: bool = True) -> typing.List[typing.Tuple[str, int]]
:canonical: src.utils.target.policy_link_updater.update_route_improvement

```{autodoc2-docstring} src.utils.target.policy_link_updater.update_route_improvement
```
````
