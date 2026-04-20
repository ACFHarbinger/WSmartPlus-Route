# {py:mod}`src.utils.validation.check_embedded_languages`

```{py:module} src.utils.validation.check_embedded_languages
```

```{autodoc2-docstring} src.utils.validation.check_embedded_languages
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EmbeddedCodeVisitor <src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_patterns <src.utils.validation.check_embedded_languages.load_patterns>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.load_patterns
    :summary:
    ```
* - {py:obj}`get_docstring_lines <src.utils.validation.check_embedded_languages.get_docstring_lines>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.get_docstring_lines
    :summary:
    ```
* - {py:obj}`detect_language <src.utils.validation.check_embedded_languages.detect_language>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.detect_language
    :summary:
    ```
* - {py:obj}`analyze_file <src.utils.validation.check_embedded_languages.analyze_file>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.analyze_file
    :summary:
    ```
* - {py:obj}`main <src.utils.validation.check_embedded_languages.main>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CYAN <src.utils.validation.check_embedded_languages.CYAN>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.CYAN
    :summary:
    ```
* - {py:obj}`GREEN <src.utils.validation.check_embedded_languages.GREEN>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.GREEN
    :summary:
    ```
* - {py:obj}`YELLOW <src.utils.validation.check_embedded_languages.YELLOW>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.YELLOW
    :summary:
    ```
* - {py:obj}`RED <src.utils.validation.check_embedded_languages.RED>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.RED
    :summary:
    ```
* - {py:obj}`RESET <src.utils.validation.check_embedded_languages.RESET>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.RESET
    :summary:
    ```
* - {py:obj}`LANGUAGE_PATTERNS <src.utils.validation.check_embedded_languages.LANGUAGE_PATTERNS>`
  - ```{autodoc2-docstring} src.utils.validation.check_embedded_languages.LANGUAGE_PATTERNS
    :summary:
    ```
````

### API

````{py:data} CYAN
:canonical: src.utils.validation.check_embedded_languages.CYAN
:value: >
   '\x1b[96m'

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.CYAN
```

````

````{py:data} GREEN
:canonical: src.utils.validation.check_embedded_languages.GREEN
:value: >
   '\x1b[92m'

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.GREEN
```

````

````{py:data} YELLOW
:canonical: src.utils.validation.check_embedded_languages.YELLOW
:value: >
   '\x1b[93m'

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.YELLOW
```

````

````{py:data} RED
:canonical: src.utils.validation.check_embedded_languages.RED
:value: >
   '\x1b[91m'

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.RED
```

````

````{py:data} RESET
:canonical: src.utils.validation.check_embedded_languages.RESET
:value: >
   '\x1b[0m'

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.RESET
```

````

````{py:function} load_patterns() -> typing.Dict[str, re.Pattern]
:canonical: src.utils.validation.check_embedded_languages.load_patterns

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.load_patterns
```
````

````{py:data} LANGUAGE_PATTERNS
:canonical: src.utils.validation.check_embedded_languages.LANGUAGE_PATTERNS
:value: >
   'load_patterns(...)'

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.LANGUAGE_PATTERNS
```

````

````{py:function} get_docstring_lines(tree: ast.AST) -> typing.Set[int]
:canonical: src.utils.validation.check_embedded_languages.get_docstring_lines

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.get_docstring_lines
```
````

````{py:function} detect_language(text: str) -> str
:canonical: src.utils.validation.check_embedded_languages.detect_language

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.detect_language
```
````

`````{py:class} EmbeddedCodeVisitor(doc_lines: typing.Set[int])
:canonical: src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor

Bases: {py:obj}`ast.NodeVisitor`

````{py:method} _check_and_record(text: str, lineno: int)
:canonical: src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor._check_and_record

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor._check_and_record
```

````

````{py:method} visit_Constant(node: ast.Constant)
:canonical: src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor.visit_Constant

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor.visit_Constant
```

````

````{py:method} visit_JoinedStr(node: ast.JoinedStr)
:canonical: src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor.visit_JoinedStr

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.EmbeddedCodeVisitor.visit_JoinedStr
```

````

`````

````{py:function} analyze_file(filepath: pathlib.Path) -> typing.List[typing.Tuple[int, str, str]]
:canonical: src.utils.validation.check_embedded_languages.analyze_file

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.analyze_file
```
````

````{py:function} main()
:canonical: src.utils.validation.check_embedded_languages.main

```{autodoc2-docstring} src.utils.validation.check_embedded_languages.main
```
````
