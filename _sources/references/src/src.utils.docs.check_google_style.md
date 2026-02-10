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
:value: >
   None

```{autodoc2-docstring} src.utils.docs.check_google_style.SECTION_ALIASES
```

````

`````{py:class} GoogleStyleValidator(filepath)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator

Bases: {py:obj}`ast.NodeVisitor`

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.__init__
```

````{py:method} add_violation(node, message)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.add_violation

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.add_violation
```

````

````{py:method} _parse_docstring_sections(docstring: str) -> typing.Dict[str, str]
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._parse_docstring_sections

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._parse_docstring_sections
```

````

````{py:method} _check_missing_args(node, sections)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._check_missing_args

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._check_missing_args
```

````

````{py:method} _check_returns_yields(node, sections)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._check_returns_yields

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._check_returns_yields
```

````

````{py:method} visit_Module(node)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_Module

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_Module
```

````

````{py:method} visit_ClassDef(node)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_ClassDef

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_ClassDef
```

````

````{py:method} visit_FunctionDef(node)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_FunctionDef

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_FunctionDef
```

````

````{py:method} visit_AsyncFunctionDef(node)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator.visit_AsyncFunctionDef

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator.visit_AsyncFunctionDef
```

````

````{py:method} _validate_function(node)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator._validate_function

```{autodoc2-docstring} src.utils.docs.check_google_style.GoogleStyleValidator._validate_function
```

````

`````

````{py:function} analyze_file(filepath)
:canonical: src.utils.docs.check_google_style.analyze_file

```{autodoc2-docstring} src.utils.docs.check_google_style.analyze_file
```
````

````{py:function} display_report(all_violations: typing.List[dict])
:canonical: src.utils.docs.check_google_style.display_report

```{autodoc2-docstring} src.utils.docs.check_google_style.display_report
```
````

````{py:function} main()
:canonical: src.utils.docs.check_google_style.main

```{autodoc2-docstring} src.utils.docs.check_google_style.main
```
````
