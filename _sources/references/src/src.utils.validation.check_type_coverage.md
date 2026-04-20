# {py:mod}`src.utils.validation.check_type_coverage`

```{py:module} src.utils.validation.check_type_coverage
```

```{autodoc2-docstring} src.utils.validation.check_type_coverage
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`analyze_function <src.utils.validation.check_type_coverage.analyze_function>`
  - ```{autodoc2-docstring} src.utils.validation.check_type_coverage.analyze_function
    :summary:
    ```
* - {py:obj}`analyze_file <src.utils.validation.check_type_coverage.analyze_file>`
  - ```{autodoc2-docstring} src.utils.validation.check_type_coverage.analyze_file
    :summary:
    ```
* - {py:obj}`_coverage_markup <src.utils.validation.check_type_coverage._coverage_markup>`
  - ```{autodoc2-docstring} src.utils.validation.check_type_coverage._coverage_markup
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.check_type_coverage.main>`
  - ```{autodoc2-docstring} src.utils.validation.check_type_coverage.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SKIP_DIRS <src.utils.validation.check_type_coverage.SKIP_DIRS>`
  - ```{autodoc2-docstring} src.utils.validation.check_type_coverage.SKIP_DIRS
    :summary:
    ```
````

### API

````{py:data} SKIP_DIRS
:canonical: src.utils.validation.check_type_coverage.SKIP_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.validation.check_type_coverage.SKIP_DIRS
```

````

````{py:function} analyze_function(node: typing.Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> typing.Tuple[int, int, bool]
:canonical: src.utils.validation.check_type_coverage.analyze_function

```{autodoc2-docstring} src.utils.validation.check_type_coverage.analyze_function
```
````

````{py:function} analyze_file(filepath: pathlib.Path) -> typing.Dict[str, int]
:canonical: src.utils.validation.check_type_coverage.analyze_file

```{autodoc2-docstring} src.utils.validation.check_type_coverage.analyze_file
```
````

````{py:function} _coverage_markup(numerator: int, denominator: int) -> str
:canonical: src.utils.validation.check_type_coverage._coverage_markup

```{autodoc2-docstring} src.utils.validation.check_type_coverage._coverage_markup
```
````

````{py:function} main() -> None
:canonical: src.utils.validation.check_type_coverage.main

```{autodoc2-docstring} src.utils.validation.check_type_coverage.main
```
````
