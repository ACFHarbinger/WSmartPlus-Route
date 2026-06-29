# {py:mod}`src.utils.package.remove_data`

```{py:module} src.utils.package.remove_data
```

```{autodoc2-docstring} src.utils.package.remove_data
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_project_root <src.utils.package.remove_data.get_project_root>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.get_project_root
    :summary:
    ```
* - {py:obj}`remove_path <src.utils.package.remove_data.remove_path>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.remove_path
    :summary:
    ```
* - {py:obj}`parse_keep_list <src.utils.package.remove_data.parse_keep_list>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.parse_keep_list
    :summary:
    ```
* - {py:obj}`clean_init_file <src.utils.package.remove_data.clean_init_file>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.clean_init_file
    :summary:
    ```
* - {py:obj}`filter_datasets <src.utils.package.remove_data.filter_datasets>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.filter_datasets
    :summary:
    ```
* - {py:obj}`filter_distributions <src.utils.package.remove_data.filter_distributions>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.filter_distributions
    :summary:
    ```
* - {py:obj}`filter_network <src.utils.package.remove_data.filter_network>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.filter_network
    :summary:
    ```
* - {py:obj}`main <src.utils.package.remove_data.main>`
  - ```{autodoc2-docstring} src.utils.package.remove_data.main
    :summary:
    ```
````

### API

````{py:function} get_project_root() -> pathlib.Path
:canonical: src.utils.package.remove_data.get_project_root

```{autodoc2-docstring} src.utils.package.remove_data.get_project_root
```
````

````{py:function} remove_path(path: pathlib.Path)
:canonical: src.utils.package.remove_data.remove_path

```{autodoc2-docstring} src.utils.package.remove_data.remove_path
```
````

````{py:function} parse_keep_list(input_str: str) -> typing.List[str]
:canonical: src.utils.package.remove_data.parse_keep_list

```{autodoc2-docstring} src.utils.package.remove_data.parse_keep_list
```
````

````{py:function} clean_init_file(init_path: pathlib.Path, deleted_module_names: typing.Set[str])
:canonical: src.utils.package.remove_data.clean_init_file

```{autodoc2-docstring} src.utils.package.remove_data.clean_init_file
```
````

````{py:function} filter_datasets(root: pathlib.Path, keep_names: typing.List[str])
:canonical: src.utils.package.remove_data.filter_datasets

```{autodoc2-docstring} src.utils.package.remove_data.filter_datasets
```
````

````{py:function} filter_distributions(root: pathlib.Path, keep_names: typing.List[str])
:canonical: src.utils.package.remove_data.filter_distributions

```{autodoc2-docstring} src.utils.package.remove_data.filter_distributions
```
````

````{py:function} filter_network(root: pathlib.Path, keep_names: typing.List[str])
:canonical: src.utils.package.remove_data.filter_network

```{autodoc2-docstring} src.utils.package.remove_data.filter_network
```
````

````{py:function} main()
:canonical: src.utils.package.remove_data.main

```{autodoc2-docstring} src.utils.package.remove_data.main
```
````
