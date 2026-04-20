# {py:mod}`src.utils.validation.check_nested_imports`

```{py:module} src.utils.validation.check_nested_imports
```

```{autodoc2-docstring} src.utils.validation.check_nested_imports
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`is_type_checking_block <src.utils.validation.check_nested_imports.is_type_checking_block>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_type_checking_block
    :summary:
    ```
* - {py:obj}`is_import_error_try_block <src.utils.validation.check_nested_imports.is_import_error_try_block>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_import_error_try_block
    :summary:
    ```
* - {py:obj}`is_suppress_import_error_block <src.utils.validation.check_nested_imports.is_suppress_import_error_block>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_suppress_import_error_block
    :summary:
    ```
* - {py:obj}`is_constant_expression <src.utils.validation.check_nested_imports.is_constant_expression>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_constant_expression
    :summary:
    ```
* - {py:obj}`is_header_assignment <src.utils.validation.check_nested_imports.is_header_assignment>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_header_assignment
    :summary:
    ```
* - {py:obj}`is_constant_guarded_if <src.utils.validation.check_nested_imports.is_constant_guarded_if>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_constant_guarded_if
    :summary:
    ```
* - {py:obj}`is_header_setup_call <src.utils.validation.check_nested_imports.is_header_setup_call>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_header_setup_call
    :summary:
    ```
* - {py:obj}`get_factory_line_ranges <src.utils.validation.check_nested_imports.get_factory_line_ranges>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.get_factory_line_ranges
    :summary:
    ```
* - {py:obj}`extract_all_imports <src.utils.validation.check_nested_imports.extract_all_imports>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.extract_all_imports
    :summary:
    ```
* - {py:obj}`analyze_file <src.utils.validation.check_nested_imports.analyze_file>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.analyze_file
    :summary:
    ```
* - {py:obj}`print_stats_table <src.utils.validation.check_nested_imports.print_stats_table>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.print_stats_table
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.check_nested_imports.main>`
  - ```{autodoc2-docstring} src.utils.validation.check_nested_imports.main
    :summary:
    ```
````

### API

````{py:function} is_type_checking_block(node: ast.stmt) -> bool
:canonical: src.utils.validation.check_nested_imports.is_type_checking_block

```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_type_checking_block
```
````

````{py:function} is_import_error_try_block(node: ast.stmt) -> bool
:canonical: src.utils.validation.check_nested_imports.is_import_error_try_block

```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_import_error_try_block
```
````

````{py:function} is_suppress_import_error_block(node: ast.stmt) -> bool
:canonical: src.utils.validation.check_nested_imports.is_suppress_import_error_block

```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_suppress_import_error_block
```
````

````{py:function} is_constant_expression(node: ast.AST) -> bool
:canonical: src.utils.validation.check_nested_imports.is_constant_expression

```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_constant_expression
```
````

````{py:function} is_header_assignment(node: ast.stmt) -> bool
:canonical: src.utils.validation.check_nested_imports.is_header_assignment

```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_header_assignment
```
````

````{py:function} is_constant_guarded_if(node: ast.stmt) -> bool
:canonical: src.utils.validation.check_nested_imports.is_constant_guarded_if

```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_constant_guarded_if
```
````

````{py:function} is_header_setup_call(node: ast.stmt) -> bool
:canonical: src.utils.validation.check_nested_imports.is_header_setup_call

```{autodoc2-docstring} src.utils.validation.check_nested_imports.is_header_setup_call
```
````

````{py:function} get_factory_line_ranges(tree: ast.AST) -> typing.List[typing.Tuple[int, int]]
:canonical: src.utils.validation.check_nested_imports.get_factory_line_ranges

```{autodoc2-docstring} src.utils.validation.check_nested_imports.get_factory_line_ranges
```
````

````{py:function} extract_all_imports(node: ast.AST) -> typing.Set[ast.AST]
:canonical: src.utils.validation.check_nested_imports.extract_all_imports

```{autodoc2-docstring} src.utils.validation.check_nested_imports.extract_all_imports
```
````

````{py:function} analyze_file(filepath: pathlib.Path, ignore_factories: bool = False) -> typing.List[typing.Tuple[int, str]]
:canonical: src.utils.validation.check_nested_imports.analyze_file

```{autodoc2-docstring} src.utils.validation.check_nested_imports.analyze_file
```
````

````{py:function} print_stats_table(all_results: typing.Dict[str, typing.List[typing.Tuple[int, str]]], target_root: pathlib.Path) -> None
:canonical: src.utils.validation.check_nested_imports.print_stats_table

```{autodoc2-docstring} src.utils.validation.check_nested_imports.print_stats_table
```
````

````{py:function} main() -> None
:canonical: src.utils.validation.check_nested_imports.main

```{autodoc2-docstring} src.utils.validation.check_nested_imports.main
```
````
