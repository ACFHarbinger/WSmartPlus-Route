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
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`analyze_file <src.utils.docs.check_google_style.analyze_file>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.analyze_file
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

* - {py:obj}`SECTION_ALIASES <src.utils.docs.check_google_style.SECTION_ALIASES>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.SECTION_ALIASES
    :summary:
    ```
* - {py:obj}`RED <src.utils.docs.check_google_style.RED>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.RED
    :summary:
    ```
* - {py:obj}`GREEN <src.utils.docs.check_google_style.GREEN>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.GREEN
    :summary:
    ```
* - {py:obj}`YELLOW <src.utils.docs.check_google_style.YELLOW>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.YELLOW
    :summary:
    ```
* - {py:obj}`CYAN <src.utils.docs.check_google_style.CYAN>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.CYAN
    :summary:
    ```
* - {py:obj}`RESET <src.utils.docs.check_google_style.RESET>`
  - ```{autodoc2-docstring} src.utils.docs.check_google_style.RESET
    :summary:
    ```
````

### API

````{py:data} SECTION_ALIASES
:canonical: src.utils.docs.check_google_style.SECTION_ALIASES
:value: >
   None

```{autodoc2-docstring} src.utils.docs.check_google_style.SECTION_ALIASES
```

````

````{py:data} RED
:canonical: src.utils.docs.check_google_style.RED
:value: >
   '\x1b[91m'

```{autodoc2-docstring} src.utils.docs.check_google_style.RED
```

````

````{py:data} GREEN
:canonical: src.utils.docs.check_google_style.GREEN
:value: >
   '\x1b[92m'

```{autodoc2-docstring} src.utils.docs.check_google_style.GREEN
```

````

````{py:data} YELLOW
:canonical: src.utils.docs.check_google_style.YELLOW
:value: >
   '\x1b[93m'

```{autodoc2-docstring} src.utils.docs.check_google_style.YELLOW
```

````

````{py:data} CYAN
:canonical: src.utils.docs.check_google_style.CYAN
:value: >
   '\x1b[96m'

```{autodoc2-docstring} src.utils.docs.check_google_style.CYAN
```

````

````{py:data} RESET
:canonical: src.utils.docs.check_google_style.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} src.utils.docs.check_google_style.RESET
```

````

`````{py:class} GoogleStyleValidator(filepath)
:canonical: src.utils.docs.check_google_style.GoogleStyleValidator

Bases: {py:obj}`ast.NodeVisitor`

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

````{py:function} main()
:canonical: src.utils.docs.check_google_style.main

```{autodoc2-docstring} src.utils.docs.check_google_style.main
```
````
