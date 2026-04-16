# {py:mod}`src.utils.io.preview`

```{py:module} src.utils.io.preview
```

```{autodoc2-docstring} src.utils.io.preview
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`preview_changes <src.utils.io.preview.preview_changes>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_changes
    :summary:
    ```
* - {py:obj}`preview_file_changes <src.utils.io.preview.preview_file_changes>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_file_changes
    :summary:
    ```
* - {py:obj}`preview_pattern_files_statistics <src.utils.io.preview.preview_pattern_files_statistics>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_pattern_files_statistics
    :summary:
    ```
* - {py:obj}`preview_file_statistics <src.utils.io.preview.preview_file_statistics>`
  - ```{autodoc2-docstring} src.utils.io.preview.preview_file_statistics
    :summary:
    ```
````

### API

````{py:function} preview_changes(root_directory: str, output_key: str = 'km', filename_pattern: str = 'log_*.json', process_func: typing.Optional[typing.Callable[..., typing.Any]] = None, update_val: typing.Union[int, float] = 0, input_keys: typing.Tuple[typing.Optional[str], typing.Optional[str]] = (None, None)) -> None
:canonical: src.utils.io.preview.preview_changes

```{autodoc2-docstring} src.utils.io.preview.preview_changes
```
````

````{py:function} preview_file_changes(file_path: str, output_key: str = 'km', process_func: typing.Optional[typing.Callable[..., typing.Any]] = None, update_val: typing.Union[int, float] = 0, input_keys: typing.Tuple[typing.Optional[str], typing.Optional[str]] = (None, None)) -> None
:canonical: src.utils.io.preview.preview_file_changes

```{autodoc2-docstring} src.utils.io.preview.preview_file_changes
```
````

````{py:function} preview_pattern_files_statistics(root_directory: str, filename_pattern: str = 'log_*.json', output_filename: str = 'output.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[..., typing.Any]] = None) -> None
:canonical: src.utils.io.preview.preview_pattern_files_statistics

```{autodoc2-docstring} src.utils.io.preview.preview_pattern_files_statistics
```
````

````{py:function} preview_file_statistics(file_path: str, output_filename: str = 'output.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[..., typing.Any]] = None) -> bool
:canonical: src.utils.io.preview.preview_file_statistics

```{autodoc2-docstring} src.utils.io.preview.preview_file_statistics
```
````
