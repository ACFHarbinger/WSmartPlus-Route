# {py:mod}`src.utils.packages.cleanup_helper`

```{py:module} src.utils.packages.cleanup_helper
```

```{autodoc2-docstring} src.utils.packages.cleanup_helper
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_project_root <src.utils.packages.cleanup_helper.get_project_root>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.get_project_root
    :summary:
    ```
* - {py:obj}`clean_init_file <src.utils.packages.cleanup_helper.clean_init_file>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.clean_init_file
    :summary:
    ```
* - {py:obj}`clean_factory_file <src.utils.packages.cleanup_helper.clean_factory_file>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.clean_factory_file
    :summary:
    ```
* - {py:obj}`remove_path <src.utils.packages.cleanup_helper.remove_path>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.remove_path
    :summary:
    ```
* - {py:obj}`_match_acronym <src.utils.packages.cleanup_helper._match_acronym>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper._match_acronym
    :summary:
    ```
* - {py:obj}`_find_yaml_to_delete <src.utils.packages.cleanup_helper._find_yaml_to_delete>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper._find_yaml_to_delete
    :summary:
    ```
* - {py:obj}`_find_configs_to_delete <src.utils.packages.cleanup_helper._find_configs_to_delete>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper._find_configs_to_delete
    :summary:
    ```
* - {py:obj}`_find_impls_to_delete <src.utils.packages.cleanup_helper._find_impls_to_delete>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper._find_impls_to_delete
    :summary:
    ```
* - {py:obj}`clean_by_acronym <src.utils.packages.cleanup_helper.clean_by_acronym>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.clean_by_acronym
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PROTECTED_DIRS <src.utils.packages.cleanup_helper.PROTECTED_DIRS>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.PROTECTED_DIRS
    :summary:
    ```
* - {py:obj}`ROUTE_CONSTRUCTORS <src.utils.packages.cleanup_helper.ROUTE_CONSTRUCTORS>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.ROUTE_CONSTRUCTORS
    :summary:
    ```
* - {py:obj}`POLICY_OTHERS <src.utils.packages.cleanup_helper.POLICY_OTHERS>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.POLICY_OTHERS
    :summary:
    ```
* - {py:obj}`ENVS <src.utils.packages.cleanup_helper.ENVS>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.ENVS
    :summary:
    ```
* - {py:obj}`MODELS <src.utils.packages.cleanup_helper.MODELS>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.MODELS
    :summary:
    ```
* - {py:obj}`RL_ALGORITHMS <src.utils.packages.cleanup_helper.RL_ALGORITHMS>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.RL_ALGORITHMS
    :summary:
    ```
* - {py:obj}`IMITATION_POLICIES <src.utils.packages.cleanup_helper.IMITATION_POLICIES>`
  - ```{autodoc2-docstring} src.utils.packages.cleanup_helper.IMITATION_POLICIES
    :summary:
    ```
````

### API

````{py:data} PROTECTED_DIRS
:canonical: src.utils.packages.cleanup_helper.PROTECTED_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.packages.cleanup_helper.PROTECTED_DIRS
```

````

````{py:data} ROUTE_CONSTRUCTORS
:canonical: src.utils.packages.cleanup_helper.ROUTE_CONSTRUCTORS
:value: >
   None

```{autodoc2-docstring} src.utils.packages.cleanup_helper.ROUTE_CONSTRUCTORS
```

````

````{py:data} POLICY_OTHERS
:canonical: src.utils.packages.cleanup_helper.POLICY_OTHERS
:value: >
   None

```{autodoc2-docstring} src.utils.packages.cleanup_helper.POLICY_OTHERS
```

````

````{py:data} ENVS
:canonical: src.utils.packages.cleanup_helper.ENVS
:value: >
   None

```{autodoc2-docstring} src.utils.packages.cleanup_helper.ENVS
```

````

````{py:data} MODELS
:canonical: src.utils.packages.cleanup_helper.MODELS
:value: >
   None

```{autodoc2-docstring} src.utils.packages.cleanup_helper.MODELS
```

````

````{py:data} RL_ALGORITHMS
:canonical: src.utils.packages.cleanup_helper.RL_ALGORITHMS
:value: >
   None

```{autodoc2-docstring} src.utils.packages.cleanup_helper.RL_ALGORITHMS
```

````

````{py:data} IMITATION_POLICIES
:canonical: src.utils.packages.cleanup_helper.IMITATION_POLICIES
:value: >
   None

```{autodoc2-docstring} src.utils.packages.cleanup_helper.IMITATION_POLICIES
```

````

````{py:function} get_project_root() -> pathlib.Path
:canonical: src.utils.packages.cleanup_helper.get_project_root

```{autodoc2-docstring} src.utils.packages.cleanup_helper.get_project_root
```
````

````{py:function} clean_init_file(init_file_path: pathlib.Path, deleted_stems: list)
:canonical: src.utils.packages.cleanup_helper.clean_init_file

```{autodoc2-docstring} src.utils.packages.cleanup_helper.clean_init_file
```
````

````{py:function} clean_factory_file(factory_file_path: pathlib.Path, deleted_stems: list)
:canonical: src.utils.packages.cleanup_helper.clean_factory_file

```{autodoc2-docstring} src.utils.packages.cleanup_helper.clean_factory_file
```
````

````{py:function} remove_path(path: pathlib.Path)
:canonical: src.utils.packages.cleanup_helper.remove_path

```{autodoc2-docstring} src.utils.packages.cleanup_helper.remove_path
```
````

````{py:function} _match_acronym(name_lower: str, acronym: str) -> bool
:canonical: src.utils.packages.cleanup_helper._match_acronym

```{autodoc2-docstring} src.utils.packages.cleanup_helper._match_acronym
```
````

````{py:function} _find_yaml_to_delete(yaml_dirs: list, acronym: typing.Optional[str], yaml_prefixes: typing.Optional[list], root: pathlib.Path, to_delete: set)
:canonical: src.utils.packages.cleanup_helper._find_yaml_to_delete

```{autodoc2-docstring} src.utils.packages.cleanup_helper._find_yaml_to_delete
```
````

````{py:function} _find_configs_to_delete(config_dirs: list, acronym: typing.Optional[str], config_prefixes: typing.Optional[list], root: pathlib.Path, to_delete: set, deleted_stems: set, affected_init_dirs: set)
:canonical: src.utils.packages.cleanup_helper._find_configs_to_delete

```{autodoc2-docstring} src.utils.packages.cleanup_helper._find_configs_to_delete
```
````

````{py:function} _find_impls_to_delete(impl_dirs: list, acronym: str, root: pathlib.Path, to_delete: set, deleted_stems: set, affected_init_dirs: set, affected_factory_dirs: set)
:canonical: src.utils.packages.cleanup_helper._find_impls_to_delete

```{autodoc2-docstring} src.utils.packages.cleanup_helper._find_impls_to_delete
```
````

````{py:function} clean_by_acronym(acronym: str, yaml_dirs: list, config_dirs: list, impl_dirs: list, yaml_prefixes: typing.Optional[list] = None, config_prefixes: typing.Optional[list] = None)
:canonical: src.utils.packages.cleanup_helper.clean_by_acronym

```{autodoc2-docstring} src.utils.packages.cleanup_helper.clean_by_acronym
```
````
