# {py:mod}`src.utils.io.statistics`

```{py:module} src.utils.io.statistics
```

```{autodoc2-docstring} src.utils.io.statistics
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`process_pattern_files_statistics <src.utils.io.statistics.process_pattern_files_statistics>`
  - ```{autodoc2-docstring} src.utils.io.statistics.process_pattern_files_statistics
    :summary:
    ```
* - {py:obj}`process_file_statistics <src.utils.io.statistics.process_file_statistics>`
  - ```{autodoc2-docstring} src.utils.io.statistics.process_file_statistics
    :summary:
    ```
````

### API

````{py:function} process_pattern_files_statistics(root_directory: str, filename_pattern: str = 'log_*.json', output_filename: str = 'output.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.List[typing.Union[int, float]]], typing.Union[int, float]]] = None) -> int
:canonical: src.utils.io.statistics.process_pattern_files_statistics

```{autodoc2-docstring} src.utils.io.statistics.process_pattern_files_statistics
```
````

````{py:function} process_file_statistics(file_path: str, output_filename: str = 'output.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.List[typing.Union[int, float]]], typing.Union[int, float]]] = None) -> bool
:canonical: src.utils.io.statistics.process_file_statistics

```{autodoc2-docstring} src.utils.io.statistics.process_file_statistics
```
````
