# {py:mod}`src.utils.docs.check_google_style`

```{py:module} src.utils.docs.check_google_style
```

```{autodoc2-docstring} src.utils.docs.check_google_style
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GoogleStyleValidator <src.utils.docs.check_google_style.GoogleStyleValidator>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`analyze_file <src.utils.docs.check_google_style.analyze_file>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.analyze_file
    :summary:
    ```
* - {py:obj}`display_report <src.utils.docs.check_google_style.display_report>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.display_report
    :summary:
    ```
* - {py:obj}`main <src.utils.docs.check_google_style.main>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`console <src.utils.docs.check_google_style.console>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.console
    :summary:
    ```
* - {py:obj}`SECTION_ALIASES <src.utils.docs.check_google_style.SECTION_ALIASES>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.SECTION_ALIASES
    :summary:
    ```
* - {py:obj}`_CONTEXT_LABEL <src.utils.docs.check_google_style._CONTEXT_LABEL>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style._CONTEXT_LABEL
    :summary:
    ```
* - {py:obj}`_CONTEXT_STYLE <src.utils.docs.check_google_style._CONTEXT_STYLE>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style._CONTEXT_STYLE
    :summary:
    ```
* - {py:obj}`SKIP_DIRS <src.utils.docs.check_google_style.SKIP_DIRS>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.SKIP_DIRS
    :summary:
    ```
````

### API

````{py:data} console
:canonical: src.utils.docs.check_google_style.console
:value: >
   'Console(...)'

```{autodoc2-docstring} src.utils.docs.check_google_style.console
```

````

````{py:data} SECTION_ALIASES
:canonical: src.utils.docs.check_google_style.SECTION_ALIASES
:type: typing.Dict[str, str]
:value: >
   None

```{autodoc2-docstring} src.utils.docs.check_google_style.SECTION_ALIASES
```

````

````{py:data} _CONTEXT_LABEL
:canonical: src.utils.docs.check_google_style._CONTEXT_LABEL
:type: typing.Dict[str, str]
:value: >
   None

```{autodoc2-docstring} src.utils.docs.check_google_style._CONTEXT_LABEL
```

````

````{py:data} _CONTEXT_STYLE
:canonical: src.utils.docs.check_google_style._CONTEXT_STYLE
:type: typing.Dict[str, str]
:value: >
   None

```{autodoc2-docstring} src.utils.docs.check_google_style._CONTEXT_STYLE
```

````

````{py:data} SKIP_DIRS
:canonical: src.utils.docs.check_google_style.SKIP_DIRS
:value: >
   None

```{autodoc2-docstring} src.utils.docs.check_google_style.SKIP_DIRS
```

````

`````{py:class} GoogleStyleValidator(filepath: str)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator

Bases: {py:obj}`ast.NodeVisitor`

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.__init__
```

````{py:method} _add(node: ast.AST, message: str) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._add

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._add
```

````

````{py:method} _parse_sections(docstring: str) -> typing.Dict[str, str]
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._parse_sections

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._parse_sections
```

````

````{py:method} _check_args(node: typing.Union[ast.FunctionDef, ast.AsyncFunctionDef], sections: typing.Dict[str, str]) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._check_args

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._check_args
```

````

````{py:method} _check_returns_yields(node: typing.Union[ast.FunctionDef, ast.AsyncFunctionDef], sections: typing.Dict[str, str]) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._check_returns_yields

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._check_returns_yields
```

````

````{py:method} visit_Module(node: ast.Module) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_Module

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_Module
```

````

````{py:method} visit_ClassDef(node: ast.ClassDef) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_ClassDef

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_ClassDef
```

````

````{py:method} visit_FunctionDef(node: ast.FunctionDef) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_FunctionDef

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_FunctionDef
```

````

````{py:method} visit_AsyncFunctionDef(node: ast.AsyncFunctionDef) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_AsyncFunctionDef

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_AsyncFunctionDef
```

````

````{py:method} _validate_function(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._validate_function

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._validate_function
```

````

`````

````{py:function} analyze_file(filepath: str) -> typing.List[dict]
:canonical: src.utils.docs.check_google_style.analyze_file

```{autodoc2-docstring} src.utils.docs.check_google_style.analyze_file
```
````

````{py:function} display_report(all_violations: typing.List[dict]) -> None
:canonical: src.utils.docs.check_google_style.display_report

```{autodoc2-docstring} src.utils.docs.check_google_style.display_report
```
````

````{py:function} main() -> None
:canonical: src.utils.docs.check_google_style.main

```{autodoc2-docstring} src.utils.docs.check_google_style.main
```
````
