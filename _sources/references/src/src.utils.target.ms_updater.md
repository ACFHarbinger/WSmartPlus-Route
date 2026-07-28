# {py:mod}`src.utils.target.ms_updater`

```{py:module} src.utils.target.ms_updater
```

```{autodoc2-docstring} src.utils.target.ms_updater
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_resolve_stem <src.utils.target.ms_updater._resolve_stem>`
  - ```{autodoc2-docstring} src.utils.target.ms_updater._resolve_stem
    :summary:
    ```
* - {py:obj}`list_available_ms_strategies <src.utils.target.ms_updater.list_available_ms_strategies>`
  - ```{autodoc2-docstring} src.utils.target.ms_updater.list_available_ms_strategies
    :summary:
    ```
* - {py:obj}`list_strategy_keys <src.utils.target.ms_updater.list_strategy_keys>`
  - ```{autodoc2-docstring} src.utils.target.ms_updater.list_strategy_keys
    :summary:
    ```
* - {py:obj}`update_mandatory_selection <src.utils.target.ms_updater.update_mandatory_selection>`
  - ```{autodoc2-docstring} src.utils.target.ms_updater.update_mandatory_selection
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SCRIPT_DIR <src.utils.target.ms_updater._SCRIPT_DIR>`
  - ```{autodoc2-docstring} src.utils.target.ms_updater._SCRIPT_DIR
    :summary:
    ```
* - {py:obj}`_DEFAULT_CONFIGS_DIR <src.utils.target.ms_updater._DEFAULT_CONFIGS_DIR>`
  - ```{autodoc2-docstring} src.utils.target.ms_updater._DEFAULT_CONFIGS_DIR
    :summary:
    ```
* - {py:obj}`_MS_FIELD_RE <src.utils.target.ms_updater._MS_FIELD_RE>`
  - ```{autodoc2-docstring} src.utils.target.ms_updater._MS_FIELD_RE
    :summary:
    ```
````

### API

````{py:data} _SCRIPT_DIR
:canonical: src.utils.target.ms_updater._SCRIPT_DIR
:value: >
   'dirname(...)'

```{autodoc2-docstring} src.utils.target.ms_updater._SCRIPT_DIR
```

````

````{py:data} _DEFAULT_CONFIGS_DIR
:canonical: src.utils.target.ms_updater._DEFAULT_CONFIGS_DIR
:value: >
   'normpath(...)'

```{autodoc2-docstring} src.utils.target.ms_updater._DEFAULT_CONFIGS_DIR
```

````

````{py:data} _MS_FIELD_RE
:canonical: src.utils.target.ms_updater._MS_FIELD_RE
:value: >
   'compile(...)'

```{autodoc2-docstring} src.utils.target.ms_updater._MS_FIELD_RE
```

````

````{py:function} _resolve_stem(yaml_arg: str) -> str
:canonical: src.utils.target.ms_updater._resolve_stem

```{autodoc2-docstring} src.utils.target.ms_updater._resolve_stem
```
````

````{py:function} list_available_ms_strategies(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.ms_updater.list_available_ms_strategies

```{autodoc2-docstring} src.utils.target.ms_updater.list_available_ms_strategies
```
````

````{py:function} list_strategy_keys(ms_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.ms_updater.list_strategy_keys

```{autodoc2-docstring} src.utils.target.ms_updater.list_strategy_keys
```
````

````{py:function} update_mandatory_selection(constructors: typing.List[str], ms_yaml: str, keys: typing.List[str], configs_dir: str = _DEFAULT_CONFIGS_DIR, dry_run: bool = False, verbose: bool = True) -> typing.List[typing.Tuple[str, int]]
:canonical: src.utils.target.ms_updater.update_mandatory_selection

```{autodoc2-docstring} src.utils.target.ms_updater.update_mandatory_selection
```
````
