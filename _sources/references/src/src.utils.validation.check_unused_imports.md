# {py:mod}`src.utils.validation.check_unused_imports`

```{py:module} src.utils.validation.check_unused_imports
```

```{autodoc2-docstring} src.utils.validation.check_unused_imports
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`UsageVisitor <src.utils.validation.check_unused_imports.UsageVisitor>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.UsageVisitor
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_factory_line_ranges <src.utils.validation.check_unused_imports.get_factory_line_ranges>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.get_factory_line_ranges
    :summary:
    ```
* - {py:obj}`analyze_file <src.utils.validation.check_unused_imports.analyze_file>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.analyze_file
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.check_unused_imports.main>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SKIP_DIRS <src.utils.validation.check_unused_imports.SKIP_DIRS>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.SKIP_DIRS
    :summary:
    ```
* - {py:obj}`RED <src.utils.validation.check_unused_imports.RED>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.RED
    :summary:
    ```
* - {py:obj}`YELLOW <src.utils.validation.check_unused_imports.YELLOW>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.YELLOW
    :summary:
    ```
* - {py:obj}`GREEN <src.utils.validation.check_unused_imports.GREEN>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.GREEN
    :summary:
    ```
* - {py:obj}`CYAN <src.utils.validation.check_unused_imports.CYAN>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.CYAN
    :summary:
    ```
* - {py:obj}`RESET <src.utils.validation.check_unused_imports.RESET>`
  - ```{autodoc2-docstring} src.utils.validation.check_unused_imports.RESET
    :summary:
    ```
````

### API

````{py:data} SKIP_DIRS
:canonical: src.utils.validation.check_unused_imports.SKIP_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.validation.check_unused_imports.SKIP_DIRS
```

````

````{py:data} RED
:canonical: src.utils.validation.check_unused_imports.RED
:value: >
   '\x1b[91m'

```{autodoc2-docstring} src.utils.validation.check_unused_imports.RED
```

````

````{py:data} YELLOW
:canonical: src.utils.validation.check_unused_imports.YELLOW
:value: >
   '\x1b[93m'

```{autodoc2-docstring} src.utils.validation.check_unused_imports.YELLOW
```

````

````{py:data} GREEN
:canonical: src.utils.validation.check_unused_imports.GREEN
:value: >
   '\x1b[92m'

```{autodoc2-docstring} src.utils.validation.check_unused_imports.GREEN
```

````

````{py:data} CYAN
:canonical: src.utils.validation.check_unused_imports.CYAN
:value: >
   '\x1b[96m'

```{autodoc2-docstring} src.utils.validation.check_unused_imports.CYAN
```

````

````{py:data} RESET
:canonical: src.utils.validation.check_unused_imports.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} src.utils.validation.check_unused_imports.RESET
```

````

`````{py:class} UsageVisitor()
:canonical: src.utils.validation.check_unused_imports.UsageVisitor

Bases: {py:obj}`ast.NodeVisitor`

```{autodoc2-docstring} src.utils.validation.check_unused_imports.UsageVisitor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.validation.check_unused_imports.UsageVisitor.__init__
```

````{py:method} visit_Name(node: ast.Name)
:canonical: src.utils.validation.check_unused_imports.UsageVisitor.visit_Name

```{autodoc2-docstring} src.utils.validation.check_unused_imports.UsageVisitor.visit_Name
```

````

````{py:method} visit_Attribute(node: ast.Attribute)
:canonical: src.utils.validation.check_unused_imports.UsageVisitor.visit_Attribute

```{autodoc2-docstring} src.utils.validation.check_unused_imports.UsageVisitor.visit_Attribute
```

````

````{py:method} visit_Constant(node: ast.Constant)
:canonical: src.utils.validation.check_unused_imports.UsageVisitor.visit_Constant

```{autodoc2-docstring} src.utils.validation.check_unused_imports.UsageVisitor.visit_Constant
```

````

`````

````{py:function} get_factory_line_ranges(tree: ast.AST) -> typing.List[typing.Tuple[int, int]]
:canonical: src.utils.validation.check_unused_imports.get_factory_line_ranges

```{autodoc2-docstring} src.utils.validation.check_unused_imports.get_factory_line_ranges
```
````

````{py:function} analyze_file(filepath: pathlib.Path, ignore_factories: bool = False) -> typing.List[typing.Tuple[int, str]]
:canonical: src.utils.validation.check_unused_imports.analyze_file

```{autodoc2-docstring} src.utils.validation.check_unused_imports.analyze_file
```
````

````{py:function} main() -> None
:canonical: src.utils.validation.check_unused_imports.main

```{autodoc2-docstring} src.utils.validation.check_unused_imports.main
```
````
