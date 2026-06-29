# {py:mod}`src.utils.target.ri_updater`

```{py:module} src.utils.target.ri_updater
```

```{autodoc2-docstring} src.utils.target.ri_updater
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_resolve_stem <src.utils.target.ri_updater._resolve_stem>`
  - ```{autodoc2-docstring} src.utils.target.ri_updater._resolve_stem
    :summary:
    ```
* - {py:obj}`list_available_ri_improvers <src.utils.target.ri_updater.list_available_ri_improvers>`
  - ```{autodoc2-docstring} src.utils.target.ri_updater.list_available_ri_improvers
    :summary:
    ```
* - {py:obj}`list_improver_keys <src.utils.target.ri_updater.list_improver_keys>`
  - ```{autodoc2-docstring} src.utils.target.ri_updater.list_improver_keys
    :summary:
    ```
* - {py:obj}`update_route_improvement <src.utils.target.ri_updater.update_route_improvement>`
  - ```{autodoc2-docstring} src.utils.target.ri_updater.update_route_improvement
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SCRIPT_DIR <src.utils.target.ri_updater._SCRIPT_DIR>`
  - ```{autodoc2-docstring} src.utils.target.ri_updater._SCRIPT_DIR
    :summary:
    ```
* - {py:obj}`_DEFAULT_CONFIGS_DIR <src.utils.target.ri_updater._DEFAULT_CONFIGS_DIR>`
  - ```{autodoc2-docstring} src.utils.target.ri_updater._DEFAULT_CONFIGS_DIR
    :summary:
    ```
* - {py:obj}`_RI_FIELD_RE <src.utils.target.ri_updater._RI_FIELD_RE>`
  - ```{autodoc2-docstring} src.utils.target.ri_updater._RI_FIELD_RE
    :summary:
    ```
````

### API

````{py:data} _SCRIPT_DIR
:canonical: src.utils.target.ri_updater._SCRIPT_DIR
:value: >
   'dirname(...)'

```{autodoc2-docstring} src.utils.target.ri_updater._SCRIPT_DIR
```

````

````{py:data} _DEFAULT_CONFIGS_DIR
:canonical: src.utils.target.ri_updater._DEFAULT_CONFIGS_DIR
:value: >
   'normpath(...)'

```{autodoc2-docstring} src.utils.target.ri_updater._DEFAULT_CONFIGS_DIR
```

````

````{py:data} _RI_FIELD_RE
:canonical: src.utils.target.ri_updater._RI_FIELD_RE
:value: >
   'compile(...)'

```{autodoc2-docstring} src.utils.target.ri_updater._RI_FIELD_RE
```

````

````{py:function} _resolve_stem(yaml_arg: str) -> str
:canonical: src.utils.target.ri_updater._resolve_stem

```{autodoc2-docstring} src.utils.target.ri_updater._resolve_stem
```
````

````{py:function} list_available_ri_improvers(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.ri_updater.list_available_ri_improvers

```{autodoc2-docstring} src.utils.target.ri_updater.list_available_ri_improvers
```
````

````{py:function} list_improver_keys(ri_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> typing.List[str]
:canonical: src.utils.target.ri_updater.list_improver_keys

```{autodoc2-docstring} src.utils.target.ri_updater.list_improver_keys
```
````

````{py:function} update_route_improvement(constructors: typing.List[str], ri_yaml: str, keys: typing.List[str], configs_dir: str = _DEFAULT_CONFIGS_DIR, dry_run: bool = False, verbose: bool = True) -> typing.List[typing.Tuple[str, int]]
:canonical: src.utils.target.ri_updater.update_route_improvement

```{autodoc2-docstring} src.utils.target.ri_updater.update_route_improvement
```
````
