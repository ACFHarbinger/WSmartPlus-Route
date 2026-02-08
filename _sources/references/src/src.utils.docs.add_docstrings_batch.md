# {py:mod}`src.utils.docs.add_docstrings_batch`

```{py:module} src.utils.docs.add_docstrings_batch
```

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DocstringInjector <src.utils.docs.add_docstrings_batch.DocstringInjector>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`main <src.utils.docs.add_docstrings_batch.main>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TEMPLATES <src.utils.docs.add_docstrings_batch.TEMPLATES>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.TEMPLATES
    :summary:
    ```
````

### API

````{py:data} TEMPLATES
:canonical: src.utils.docs.add_docstrings_batch.TEMPLATES
:value: >
   None

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.TEMPLATES
```

````

`````{py:class} DocstringInjector(filepath: str)
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector.__init__
```

````{py:method} _get_indent(lineno: int) -> str
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector._get_indent

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector._get_indent
```

````

````{py:method} _format_args(args: typing.List[ast.arg], base_indent: str) -> str
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector._format_args

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector._format_args
```

````

````{py:method} generate_docstring(node: typing.Any, context: str = '') -> str
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector.generate_docstring

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector.generate_docstring
```

````

````{py:method} _find_insertion_line(node: typing.Any) -> int
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector._find_insertion_line

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector._find_insertion_line
```

````

````{py:method} scan_and_queue()
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector.scan_and_queue

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector.scan_and_queue
```

````

````{py:method} apply()
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector.apply

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector.apply
```

````

````{py:method} save()
:canonical: src.utils.docs.add_docstrings_batch.DocstringInjector.save

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DocstringInjector.save
```

````

`````

````{py:function} main()
:canonical: src.utils.docs.add_docstrings_batch.main

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.main
```
````
