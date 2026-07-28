# {py:mod}`src.utils.validation.check_interface_compliance`

```{py:module} src.utils.validation.check_interface_compliance
```

```{autodoc2-docstring} src.utils.validation.check_interface_compliance
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`collect_python_files <src.utils.validation.check_interface_compliance.collect_python_files>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.collect_python_files
    :summary:
    ```
* - {py:obj}`base_name <src.utils.validation.check_interface_compliance.base_name>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.base_name
    :summary:
    ```
* - {py:obj}`has_decorator <src.utils.validation.check_interface_compliance.has_decorator>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.has_decorator
    :summary:
    ```
* - {py:obj}`parse_file <src.utils.validation.check_interface_compliance.parse_file>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.parse_file
    :summary:
    ```
* - {py:obj}`build_registry <src.utils.validation.check_interface_compliance.build_registry>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.build_registry
    :summary:
    ```
* - {py:obj}`get_required_abstract_methods <src.utils.validation.check_interface_compliance.get_required_abstract_methods>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.get_required_abstract_methods
    :summary:
    ```
* - {py:obj}`get_implemented_methods <src.utils.validation.check_interface_compliance.get_implemented_methods>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.get_implemented_methods
    :summary:
    ```
* - {py:obj}`check_compliance <src.utils.validation.check_interface_compliance.check_compliance>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.check_compliance
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.check_interface_compliance.main>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SKIP_DIRS <src.utils.validation.check_interface_compliance.SKIP_DIRS>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.SKIP_DIRS
    :summary:
    ```
* - {py:obj}`RED <src.utils.validation.check_interface_compliance.RED>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.RED
    :summary:
    ```
* - {py:obj}`YELLOW <src.utils.validation.check_interface_compliance.YELLOW>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.YELLOW
    :summary:
    ```
* - {py:obj}`GREEN <src.utils.validation.check_interface_compliance.GREEN>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.GREEN
    :summary:
    ```
* - {py:obj}`CYAN <src.utils.validation.check_interface_compliance.CYAN>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.CYAN
    :summary:
    ```
* - {py:obj}`BOLD <src.utils.validation.check_interface_compliance.BOLD>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.BOLD
    :summary:
    ```
* - {py:obj}`DIM <src.utils.validation.check_interface_compliance.DIM>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.DIM
    :summary:
    ```
* - {py:obj}`RESET <src.utils.validation.check_interface_compliance.RESET>`
  - ```{autodoc2-docstring} src.utils.validation.check_interface_compliance.RESET
    :summary:
    ```
````

### API

````{py:data} SKIP_DIRS
:canonical: src.utils.validation.check_interface_compliance.SKIP_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.SKIP_DIRS
```

````

````{py:data} RED
:canonical: src.utils.validation.check_interface_compliance.RED
:value: >
   '\x1b[91m'

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.RED
```

````

````{py:data} YELLOW
:canonical: src.utils.validation.check_interface_compliance.YELLOW
:value: >
   '\x1b[93m'

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.YELLOW
```

````

````{py:data} GREEN
:canonical: src.utils.validation.check_interface_compliance.GREEN
:value: >
   '\x1b[92m'

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.GREEN
```

````

````{py:data} CYAN
:canonical: src.utils.validation.check_interface_compliance.CYAN
:value: >
   '\x1b[96m'

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.CYAN
```

````

````{py:data} BOLD
:canonical: src.utils.validation.check_interface_compliance.BOLD
:value: >
   '\x1b[1m'

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.BOLD
```

````

````{py:data} DIM
:canonical: src.utils.validation.check_interface_compliance.DIM
:value: >
   '\x1b[2m'

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.DIM
```

````

````{py:data} RESET
:canonical: src.utils.validation.check_interface_compliance.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.RESET
```

````

````{py:function} collect_python_files(root: pathlib.Path, exclude: typing.Set[str]) -> typing.List[pathlib.Path]
:canonical: src.utils.validation.check_interface_compliance.collect_python_files

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.collect_python_files
```
````

````{py:function} base_name(base_node: ast.expr) -> str
:canonical: src.utils.validation.check_interface_compliance.base_name

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.base_name
```
````

````{py:function} has_decorator(func_node: ast.FunctionDef, name: str) -> bool
:canonical: src.utils.validation.check_interface_compliance.has_decorator

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.has_decorator
```
````

````{py:function} parse_file(filepath: pathlib.Path) -> typing.List[typing.Dict]
:canonical: src.utils.validation.check_interface_compliance.parse_file

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.parse_file
```
````

````{py:function} build_registry(files: typing.List[pathlib.Path]) -> typing.Dict[str, typing.List[typing.Dict]]
:canonical: src.utils.validation.check_interface_compliance.build_registry

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.build_registry
```
````

````{py:function} get_required_abstract_methods(class_name: str, registry: typing.Dict[str, typing.List[typing.Dict]], _seen: typing.Optional[typing.Set[str]] = None) -> typing.Set[str]
:canonical: src.utils.validation.check_interface_compliance.get_required_abstract_methods

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.get_required_abstract_methods
```
````

````{py:function} get_implemented_methods(class_name: str, registry: typing.Dict[str, typing.List[typing.Dict]], _seen: typing.Optional[typing.Set[str]] = None) -> typing.Set[str]
:canonical: src.utils.validation.check_interface_compliance.get_implemented_methods

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.get_implemented_methods
```
````

````{py:function} check_compliance(registry: typing.Dict[str, typing.List[typing.Dict]]) -> typing.List[typing.Dict]
:canonical: src.utils.validation.check_interface_compliance.check_compliance

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.check_compliance
```
````

````{py:function} main() -> None
:canonical: src.utils.validation.check_interface_compliance.main

```{autodoc2-docstring} src.utils.validation.check_interface_compliance.main
```
````
