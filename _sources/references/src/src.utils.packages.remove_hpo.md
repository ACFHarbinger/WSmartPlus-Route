# {py:mod}`src.utils.packages.remove_hpo`

```{py:module} src.utils.packages.remove_hpo
```

```{autodoc2-docstring} src.utils.packages.remove_hpo
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_project_root <src.utils.packages.remove_hpo.get_project_root>`
  - ```{autodoc2-docstring} src.utils.packages.remove_hpo.get_project_root
    :summary:
    ```
* - {py:obj}`remove_path <src.utils.packages.remove_hpo.remove_path>`
  - ```{autodoc2-docstring} src.utils.packages.remove_hpo.remove_path
    :summary:
    ```
* - {py:obj}`clean_tasks_init <src.utils.packages.remove_hpo.clean_tasks_init>`
  - ```{autodoc2-docstring} src.utils.packages.remove_hpo.clean_tasks_init
    :summary:
    ```
* - {py:obj}`clean_configs_init <src.utils.packages.remove_hpo.clean_configs_init>`
  - ```{autodoc2-docstring} src.utils.packages.remove_hpo.clean_configs_init
    :summary:
    ```
* - {py:obj}`remove_constants_init_import <src.utils.packages.remove_hpo.remove_constants_init_import>`
  - ```{autodoc2-docstring} src.utils.packages.remove_hpo.remove_constants_init_import
    :summary:
    ```
* - {py:obj}`clean_hydra_dispatch <src.utils.packages.remove_hpo.clean_hydra_dispatch>`
  - ```{autodoc2-docstring} src.utils.packages.remove_hpo.clean_hydra_dispatch
    :summary:
    ```
* - {py:obj}`main <src.utils.packages.remove_hpo.main>`
  - ```{autodoc2-docstring} src.utils.packages.remove_hpo.main
    :summary:
    ```
````

### API

````{py:function} get_project_root() -> pathlib.Path
:canonical: src.utils.packages.remove_hpo.get_project_root

```{autodoc2-docstring} src.utils.packages.remove_hpo.get_project_root
```
````

````{py:function} remove_path(path: pathlib.Path) -> None
:canonical: src.utils.packages.remove_hpo.remove_path

```{autodoc2-docstring} src.utils.packages.remove_hpo.remove_path
```
````

````{py:function} clean_tasks_init(init_path: pathlib.Path, class_names: list) -> None
:canonical: src.utils.packages.remove_hpo.clean_tasks_init

```{autodoc2-docstring} src.utils.packages.remove_hpo.clean_tasks_init
```
````

````{py:function} clean_configs_init(init_path: pathlib.Path, class_names: list, field_names: list) -> None
:canonical: src.utils.packages.remove_hpo.clean_configs_init

```{autodoc2-docstring} src.utils.packages.remove_hpo.clean_configs_init
```
````

````{py:function} remove_constants_init_import(root: pathlib.Path, module_name: str) -> None
:canonical: src.utils.packages.remove_hpo.remove_constants_init_import

```{autodoc2-docstring} src.utils.packages.remove_hpo.remove_constants_init_import
```
````

````{py:function} clean_hydra_dispatch(dispatch_path: pathlib.Path) -> None
:canonical: src.utils.packages.remove_hpo.clean_hydra_dispatch

```{autodoc2-docstring} src.utils.packages.remove_hpo.clean_hydra_dispatch
```
````

````{py:function} main() -> None
:canonical: src.utils.packages.remove_hpo.main

```{autodoc2-docstring} src.utils.packages.remove_hpo.main
```
````
