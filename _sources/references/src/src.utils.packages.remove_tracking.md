# {py:mod}`src.utils.packages.remove_tracking`

```{py:module} src.utils.packages.remove_tracking
```

```{autodoc2-docstring} src.utils.packages.remove_tracking
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_project_root <src.utils.packages.remove_tracking.get_project_root>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.get_project_root
    :summary:
    ```
* - {py:obj}`remove_path <src.utils.packages.remove_tracking.remove_path>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.remove_path
    :summary:
    ```
* - {py:obj}`fix_empty_try_blocks <src.utils.packages.remove_tracking.fix_empty_try_blocks>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.fix_empty_try_blocks
    :summary:
    ```
* - {py:obj}`append_to_class_body <src.utils.packages.remove_tracking.append_to_class_body>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.append_to_class_body
    :summary:
    ```
* - {py:obj}`remove_viz_mixin_from_file <src.utils.packages.remove_tracking.remove_viz_mixin_from_file>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.remove_viz_mixin_from_file
    :summary:
    ```
* - {py:obj}`patch_specific_files <src.utils.packages.remove_tracking.patch_specific_files>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.patch_specific_files
    :summary:
    ```
* - {py:obj}`comment_tracking_imports <src.utils.packages.remove_tracking.comment_tracking_imports>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.comment_tracking_imports
    :summary:
    ```
* - {py:obj}`replace_logger_calls_with_print <src.utils.packages.remove_tracking.replace_logger_calls_with_print>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.replace_logger_calls_with_print
    :summary:
    ```
* - {py:obj}`redirect_logging_imports <src.utils.packages.remove_tracking.redirect_logging_imports>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.redirect_logging_imports
    :summary:
    ```
* - {py:obj}`comment_justfile <src.utils.packages.remove_tracking.comment_justfile>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.comment_justfile
    :summary:
    ```
* - {py:obj}`clean_test_functions <src.utils.packages.remove_tracking.clean_test_functions>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.clean_test_functions
    :summary:
    ```
* - {py:obj}`main <src.utils.packages.remove_tracking.main>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_LOG_UTILS_STUBS <src.utils.packages.remove_tracking._LOG_UTILS_STUBS>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking._LOG_UTILS_STUBS
    :summary:
    ```
* - {py:obj}`_LOGGER_WRITER_STUBS <src.utils.packages.remove_tracking._LOGGER_WRITER_STUBS>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking._LOGGER_WRITER_STUBS
    :summary:
    ```
* - {py:obj}`_SPECIFIC_PATCH_NAMES <src.utils.packages.remove_tracking._SPECIFIC_PATCH_NAMES>`
  - ```{autodoc2-docstring} src.utils.packages.remove_tracking._SPECIFIC_PATCH_NAMES
    :summary:
    ```
````

### API

````{py:function} get_project_root() -> pathlib.Path
:canonical: src.utils.packages.remove_tracking.get_project_root

```{autodoc2-docstring} src.utils.packages.remove_tracking.get_project_root
```
````

````{py:function} remove_path(path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.remove_path

```{autodoc2-docstring} src.utils.packages.remove_tracking.remove_path
```
````

````{py:function} fix_empty_try_blocks(content: str) -> str
:canonical: src.utils.packages.remove_tracking.fix_empty_try_blocks

```{autodoc2-docstring} src.utils.packages.remove_tracking.fix_empty_try_blocks
```
````

````{py:function} append_to_class_body(content: str, class_name: str, method_lines: str) -> str
:canonical: src.utils.packages.remove_tracking.append_to_class_body

```{autodoc2-docstring} src.utils.packages.remove_tracking.append_to_class_body
```
````

````{py:function} remove_viz_mixin_from_file(file_path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.remove_viz_mixin_from_file

```{autodoc2-docstring} src.utils.packages.remove_tracking.remove_viz_mixin_from_file
```
````

````{py:function} patch_specific_files(file_path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.patch_specific_files

```{autodoc2-docstring} src.utils.packages.remove_tracking.patch_specific_files
```
````

````{py:function} comment_tracking_imports(file_path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.comment_tracking_imports

```{autodoc2-docstring} src.utils.packages.remove_tracking.comment_tracking_imports
```
````

````{py:data} _LOG_UTILS_STUBS
:canonical: src.utils.packages.remove_tracking._LOG_UTILS_STUBS
:value: <Multiline-String>

```{autodoc2-docstring} src.utils.packages.remove_tracking._LOG_UTILS_STUBS
```

````

````{py:data} _LOGGER_WRITER_STUBS
:canonical: src.utils.packages.remove_tracking._LOGGER_WRITER_STUBS
:value: <Multiline-String>

```{autodoc2-docstring} src.utils.packages.remove_tracking._LOGGER_WRITER_STUBS
```

````

````{py:function} replace_logger_calls_with_print(file_path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.replace_logger_calls_with_print

```{autodoc2-docstring} src.utils.packages.remove_tracking.replace_logger_calls_with_print
```
````

````{py:function} redirect_logging_imports(file_path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.redirect_logging_imports

```{autodoc2-docstring} src.utils.packages.remove_tracking.redirect_logging_imports
```
````

````{py:function} comment_justfile(justfile_path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.comment_justfile

```{autodoc2-docstring} src.utils.packages.remove_tracking.comment_justfile
```
````

````{py:function} clean_test_functions(file_path: pathlib.Path)
:canonical: src.utils.packages.remove_tracking.clean_test_functions

```{autodoc2-docstring} src.utils.packages.remove_tracking.clean_test_functions
```
````

````{py:data} _SPECIFIC_PATCH_NAMES
:canonical: src.utils.packages.remove_tracking._SPECIFIC_PATCH_NAMES
:value: >
   'frozenset(...)'

```{autodoc2-docstring} src.utils.packages.remove_tracking._SPECIFIC_PATCH_NAMES
```

````

````{py:function} main()
:canonical: src.utils.packages.remove_tracking.main

```{autodoc2-docstring} src.utils.packages.remove_tracking.main
```
````
