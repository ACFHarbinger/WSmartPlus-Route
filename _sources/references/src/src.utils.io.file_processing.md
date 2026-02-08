# {py:mod}`src.utils.io.file_processing`

```{py:module} src.utils.io.file_processing
```

```{autodoc2-docstring} src.utils.io.file_processing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`process_pattern_files <src.utils.io.file_processing.process_pattern_files>`
  - ```{autodoc2-docstring} src.utils.io.file_processing.process_pattern_files
    :summary:
    ```
* - {py:obj}`process_file <src.utils.io.file_processing.process_file>`
  - ```{autodoc2-docstring} src.utils.io.file_processing.process_file
    :summary:
    ```
````

### API

````{py:function} process_pattern_files(root_directory: str, filename_pattern: str = 'log_*.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0, input_keys: typing.Tuple[typing.Optional[str], typing.Union[str, int, float, None]] = (None, None)) -> int
:canonical: src.utils.io.file_processing.process_pattern_files

```{autodoc2-docstring} src.utils.io.file_processing.process_pattern_files
```
````

````{py:function} process_file(file_path: str, output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0, input_keys: typing.Tuple[typing.Optional[str], typing.Union[str, int, float, None]] = (None, None)) -> bool
:canonical: src.utils.io.file_processing.process_file

```{autodoc2-docstring} src.utils.io.file_processing.process_file
```
````
