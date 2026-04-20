# {py:mod}`src.utils.validation.check_relative_imports`

```{py:module} src.utils.validation.check_relative_imports
```

```{autodoc2-docstring} src.utils.validation.check_relative_imports
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`format_relative_import <src.utils.validation.check_relative_imports.format_relative_import>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.format_relative_import
    :summary:
    ```
* - {py:obj}`analyze_file <src.utils.validation.check_relative_imports.analyze_file>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.analyze_file
    :summary:
    ```
* - {py:obj}`print_stats_table <src.utils.validation.check_relative_imports.print_stats_table>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.print_stats_table
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.check_relative_imports.main>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SKIP_DIRS <src.utils.validation.check_relative_imports.SKIP_DIRS>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.SKIP_DIRS
    :summary:
    ```
* - {py:obj}`RED <src.utils.validation.check_relative_imports.RED>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.RED
    :summary:
    ```
* - {py:obj}`YELLOW <src.utils.validation.check_relative_imports.YELLOW>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.YELLOW
    :summary:
    ```
* - {py:obj}`GREEN <src.utils.validation.check_relative_imports.GREEN>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.GREEN
    :summary:
    ```
* - {py:obj}`CYAN <src.utils.validation.check_relative_imports.CYAN>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.CYAN
    :summary:
    ```
* - {py:obj}`DIM <src.utils.validation.check_relative_imports.DIM>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.DIM
    :summary:
    ```
* - {py:obj}`RESET <src.utils.validation.check_relative_imports.RESET>`
  - ```{autodoc2-docstring} src.utils.validation.check_relative_imports.RESET
    :summary:
    ```
````

### API

````{py:data} SKIP_DIRS
:canonical: src.utils.validation.check_relative_imports.SKIP_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.validation.check_relative_imports.SKIP_DIRS
```

````

````{py:data} RED
:canonical: src.utils.validation.check_relative_imports.RED
:value: >
   '\x1b[91m'

```{autodoc2-docstring} src.utils.validation.check_relative_imports.RED
```

````

````{py:data} YELLOW
:canonical: src.utils.validation.check_relative_imports.YELLOW
:value: >
   '\x1b[93m'

```{autodoc2-docstring} src.utils.validation.check_relative_imports.YELLOW
```

````

````{py:data} GREEN
:canonical: src.utils.validation.check_relative_imports.GREEN
:value: >
   '\x1b[92m'

```{autodoc2-docstring} src.utils.validation.check_relative_imports.GREEN
```

````

````{py:data} CYAN
:canonical: src.utils.validation.check_relative_imports.CYAN
:value: >
   '\x1b[96m'

```{autodoc2-docstring} src.utils.validation.check_relative_imports.CYAN
```

````

````{py:data} DIM
:canonical: src.utils.validation.check_relative_imports.DIM
:value: >
   '\x1b[2m'

```{autodoc2-docstring} src.utils.validation.check_relative_imports.DIM
```

````

````{py:data} RESET
:canonical: src.utils.validation.check_relative_imports.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} src.utils.validation.check_relative_imports.RESET
```

````

````{py:function} format_relative_import(node: ast.ImportFrom) -> str
:canonical: src.utils.validation.check_relative_imports.format_relative_import

```{autodoc2-docstring} src.utils.validation.check_relative_imports.format_relative_import
```
````

````{py:function} analyze_file(filepath: pathlib.Path) -> typing.List[typing.Tuple[int, int, str]]
:canonical: src.utils.validation.check_relative_imports.analyze_file

```{autodoc2-docstring} src.utils.validation.check_relative_imports.analyze_file
```
````

````{py:function} print_stats_table(all_results: typing.Dict[str, typing.List[typing.Tuple[int, int, str]]], target_root: pathlib.Path) -> None
:canonical: src.utils.validation.check_relative_imports.print_stats_table

```{autodoc2-docstring} src.utils.validation.check_relative_imports.print_stats_table
```
````

````{py:function} main() -> None
:canonical: src.utils.validation.check_relative_imports.main

```{autodoc2-docstring} src.utils.validation.check_relative_imports.main
```
````
