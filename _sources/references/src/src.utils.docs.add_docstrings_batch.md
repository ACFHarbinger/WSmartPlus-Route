# {py:mod}`src.utils.docs.add_docstrings_batch`

```{py:module} src.utils.docs.add_docstrings_batch
```

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`generate_docstring <src.utils.docs.add_docstrings_batch.generate_docstring>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.generate_docstring
    :summary:
    ```
* - {py:obj}`add_docstring_to_function <src.utils.docs.add_docstrings_batch.add_docstring_to_function>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.add_docstring_to_function
    :summary:
    ```
* - {py:obj}`process_file <src.utils.docs.add_docstrings_batch.process_file>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.process_file
    :summary:
    ```
* - {py:obj}`main <src.utils.docs.add_docstrings_batch.main>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MISSING_DOCSTRINGS <src.utils.docs.add_docstrings_batch.MISSING_DOCSTRINGS>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.MISSING_DOCSTRINGS
    :summary:
    ```
* - {py:obj}`DOCSTRING_TEMPLATES <src.utils.docs.add_docstrings_batch.DOCSTRING_TEMPLATES>`
  - ```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DOCSTRING_TEMPLATES
    :summary:
    ```
````

### API

````{py:data} MISSING_DOCSTRINGS
:canonical: src.utils.docs.add_docstrings_batch.MISSING_DOCSTRINGS
:value: >
   None

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.MISSING_DOCSTRINGS
```

````

````{py:data} DOCSTRING_TEMPLATES
:canonical: src.utils.docs.add_docstrings_batch.DOCSTRING_TEMPLATES
:value: >
   None

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.DOCSTRING_TEMPLATES
```

````

````{py:function} generate_docstring(func_name: str, class_name: str, args_list: typing.List[str]) -> str
:canonical: src.utils.docs.add_docstrings_batch.generate_docstring

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.generate_docstring
```
````

````{py:function} add_docstring_to_function(content: str, func_name: str, class_name: str = '') -> str
:canonical: src.utils.docs.add_docstrings_batch.add_docstring_to_function

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.add_docstring_to_function
```
````

````{py:function} process_file(file_path: str, missing_funcs: typing.List[str])
:canonical: src.utils.docs.add_docstrings_batch.process_file

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.process_file
```
````

````{py:function} main()
:canonical: src.utils.docs.add_docstrings_batch.main

```{autodoc2-docstring} src.utils.docs.add_docstrings_batch.main
```
````
