# {py:mod}`src.utils.validation.check_circular_imports`

```{py:module} src.utils.validation.check_circular_imports
```

```{autodoc2-docstring} src.utils.validation.check_circular_imports
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImportVisitor <src.utils.validation.check_circular_imports.ImportVisitor>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`file_to_module <src.utils.validation.check_circular_imports.file_to_module>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.file_to_module
    :summary:
    ```
* - {py:obj}`collect_module_map <src.utils.validation.check_circular_imports.collect_module_map>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.collect_module_map
    :summary:
    ```
* - {py:obj}`resolve_to_module <src.utils.validation.check_circular_imports.resolve_to_module>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.resolve_to_module
    :summary:
    ```
* - {py:obj}`build_graph <src.utils.validation.check_circular_imports.build_graph>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.build_graph
    :summary:
    ```
* - {py:obj}`tarjan_sccs <src.utils.validation.check_circular_imports.tarjan_sccs>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.tarjan_sccs
    :summary:
    ```
* - {py:obj}`generate_html <src.utils.validation.check_circular_imports.generate_html>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.generate_html
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.check_circular_imports.main>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SKIP_DIRS <src.utils.validation.check_circular_imports.SKIP_DIRS>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.SKIP_DIRS
    :summary:
    ```
* - {py:obj}`RED <src.utils.validation.check_circular_imports.RED>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.RED
    :summary:
    ```
* - {py:obj}`YELLOW <src.utils.validation.check_circular_imports.YELLOW>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.YELLOW
    :summary:
    ```
* - {py:obj}`GREEN <src.utils.validation.check_circular_imports.GREEN>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.GREEN
    :summary:
    ```
* - {py:obj}`CYAN <src.utils.validation.check_circular_imports.CYAN>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.CYAN
    :summary:
    ```
* - {py:obj}`BOLD <src.utils.validation.check_circular_imports.BOLD>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.BOLD
    :summary:
    ```
* - {py:obj}`DIM <src.utils.validation.check_circular_imports.DIM>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.DIM
    :summary:
    ```
* - {py:obj}`RESET <src.utils.validation.check_circular_imports.RESET>`
  - ```{autodoc2-docstring} src.utils.validation.check_circular_imports.RESET
    :summary:
    ```
````

### API

````{py:data} SKIP_DIRS
:canonical: src.utils.validation.check_circular_imports.SKIP_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.validation.check_circular_imports.SKIP_DIRS
```

````

````{py:data} RED
:canonical: src.utils.validation.check_circular_imports.RED
:value: >
   '\x1b[91m'

```{autodoc2-docstring} src.utils.validation.check_circular_imports.RED
```

````

````{py:data} YELLOW
:canonical: src.utils.validation.check_circular_imports.YELLOW
:value: >
   '\x1b[93m'

```{autodoc2-docstring} src.utils.validation.check_circular_imports.YELLOW
```

````

````{py:data} GREEN
:canonical: src.utils.validation.check_circular_imports.GREEN
:value: >
   '\x1b[92m'

```{autodoc2-docstring} src.utils.validation.check_circular_imports.GREEN
```

````

````{py:data} CYAN
:canonical: src.utils.validation.check_circular_imports.CYAN
:value: >
   '\x1b[96m'

```{autodoc2-docstring} src.utils.validation.check_circular_imports.CYAN
```

````

````{py:data} BOLD
:canonical: src.utils.validation.check_circular_imports.BOLD
:value: >
   '\x1b[1m'

```{autodoc2-docstring} src.utils.validation.check_circular_imports.BOLD
```

````

````{py:data} DIM
:canonical: src.utils.validation.check_circular_imports.DIM
:value: >
   '\x1b[2m'

```{autodoc2-docstring} src.utils.validation.check_circular_imports.DIM
```

````

````{py:data} RESET
:canonical: src.utils.validation.check_circular_imports.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} src.utils.validation.check_circular_imports.RESET
```

````

````{py:function} file_to_module(filepath: pathlib.Path, root: pathlib.Path) -> str
:canonical: src.utils.validation.check_circular_imports.file_to_module

```{autodoc2-docstring} src.utils.validation.check_circular_imports.file_to_module
```
````

````{py:function} collect_module_map(root: pathlib.Path, exclude: typing.Set[str]) -> typing.Dict[str, pathlib.Path]
:canonical: src.utils.validation.check_circular_imports.collect_module_map

```{autodoc2-docstring} src.utils.validation.check_circular_imports.collect_module_map
```
````

````{py:function} resolve_to_module(raw: str, level: int, current: str, known: typing.Set[str]) -> typing.Optional[str]
:canonical: src.utils.validation.check_circular_imports.resolve_to_module

```{autodoc2-docstring} src.utils.validation.check_circular_imports.resolve_to_module
```
````

`````{py:class} ImportVisitor(module: str, known: typing.Set[str])
:canonical: src.utils.validation.check_circular_imports.ImportVisitor

Bases: {py:obj}`ast.NodeVisitor`

```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor.__init__
```

````{py:method} visit_Import(node: ast.Import)
:canonical: src.utils.validation.check_circular_imports.ImportVisitor.visit_Import

```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor.visit_Import
```

````

````{py:method} visit_ImportFrom(node: ast.ImportFrom)
:canonical: src.utils.validation.check_circular_imports.ImportVisitor.visit_ImportFrom

```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor.visit_ImportFrom
```

````

````{py:method} visit_If(node: ast.If)
:canonical: src.utils.validation.check_circular_imports.ImportVisitor.visit_If

```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor.visit_If
```

````

````{py:method} visit_FunctionDef(node: ast.FunctionDef)
:canonical: src.utils.validation.check_circular_imports.ImportVisitor.visit_FunctionDef

```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor.visit_FunctionDef
```

````

````{py:method} visit_AsyncFunctionDef(node: ast.AsyncFunctionDef)
:canonical: src.utils.validation.check_circular_imports.ImportVisitor.visit_AsyncFunctionDef

```{autodoc2-docstring} src.utils.validation.check_circular_imports.ImportVisitor.visit_AsyncFunctionDef
```

````

`````

````{py:function} build_graph(root: pathlib.Path, exclude: typing.Set[str]) -> typing.Dict[str, typing.Set[str]]
:canonical: src.utils.validation.check_circular_imports.build_graph

```{autodoc2-docstring} src.utils.validation.check_circular_imports.build_graph
```
````

````{py:function} tarjan_sccs(graph: typing.Dict[str, typing.Set[str]]) -> typing.List[typing.List[str]]
:canonical: src.utils.validation.check_circular_imports.tarjan_sccs

```{autodoc2-docstring} src.utils.validation.check_circular_imports.tarjan_sccs
```
````

````{py:function} generate_html(cycles: typing.List[typing.List[str]], graph: typing.Dict[str, typing.Set[str]], output: pathlib.Path) -> None
:canonical: src.utils.validation.check_circular_imports.generate_html

```{autodoc2-docstring} src.utils.validation.check_circular_imports.generate_html
```
````

````{py:function} main() -> None
:canonical: src.utils.validation.check_circular_imports.main

```{autodoc2-docstring} src.utils.validation.check_circular_imports.main
```
````
