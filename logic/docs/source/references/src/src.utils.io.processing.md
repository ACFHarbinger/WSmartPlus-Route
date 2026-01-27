# {py:mod}`src.utils.io.processing`

```{py:module} src.utils.io.processing
```

```{autodoc2-docstring} src.utils.io.processing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`process_dict_of_dicts <src.utils.io.processing.process_dict_of_dicts>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_dict_of_dicts
    :summary:
    ```
* - {py:obj}`process_list_of_dicts <src.utils.io.processing.process_list_of_dicts>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_list_of_dicts
    :summary:
    ```
* - {py:obj}`process_dict_two_inputs <src.utils.io.processing.process_dict_two_inputs>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_dict_two_inputs
    :summary:
    ```
* - {py:obj}`process_list_two_inputs <src.utils.io.processing.process_list_two_inputs>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_list_two_inputs
    :summary:
    ```
* - {py:obj}`find_single_input_values <src.utils.io.processing.find_single_input_values>`
  - ```{autodoc2-docstring} src.utils.io.processing.find_single_input_values
    :summary:
    ```
* - {py:obj}`find_two_input_values <src.utils.io.processing.find_two_input_values>`
  - ```{autodoc2-docstring} src.utils.io.processing.find_two_input_values
    :summary:
    ```
* - {py:obj}`process_pattern_files <src.utils.io.processing.process_pattern_files>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_pattern_files
    :summary:
    ```
* - {py:obj}`process_file <src.utils.io.processing.process_file>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_file
    :summary:
    ```
* - {py:obj}`process_pattern_files_statistics <src.utils.io.processing.process_pattern_files_statistics>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_pattern_files_statistics
    :summary:
    ```
* - {py:obj}`process_file_statistics <src.utils.io.processing.process_file_statistics>`
  - ```{autodoc2-docstring} src.utils.io.processing.process_file_statistics
    :summary:
    ```
````

### API

````{py:function} process_dict_of_dicts(data_dict: typing.Dict[str, typing.Any], output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0) -> bool
:canonical: src.utils.io.processing.process_dict_of_dicts

```{autodoc2-docstring} src.utils.io.processing.process_dict_of_dicts
```
````

````{py:function} process_list_of_dicts(data_list: typing.List[typing.Dict[str, typing.Any]], output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0) -> bool
:canonical: src.utils.io.processing.process_list_of_dicts

```{autodoc2-docstring} src.utils.io.processing.process_list_of_dicts
```
````

````{py:function} process_dict_two_inputs(data_dict: typing.Dict[str, typing.Any], input_key1: str, input_key2_or_val: typing.Union[str, int, float], output_key: str, process_func: typing.Callable[[typing.Any, typing.Any], typing.Any]) -> bool
:canonical: src.utils.io.processing.process_dict_two_inputs

```{autodoc2-docstring} src.utils.io.processing.process_dict_two_inputs
```
````

````{py:function} process_list_two_inputs(data_list: typing.List[typing.Dict[str, typing.Any]], input_key1: str, input_key2_or_val: typing.Union[str, int, float], output_key: str, process_func: typing.Callable[[typing.Any, typing.Any], typing.Any]) -> bool
:canonical: src.utils.io.processing.process_list_two_inputs

```{autodoc2-docstring} src.utils.io.processing.process_list_two_inputs
```
````

````{py:function} find_single_input_values(data: typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]], current_path: str = '', output_key: str = 'km') -> typing.List[typing.Tuple[str, typing.Union[int, float]]]
:canonical: src.utils.io.processing.find_single_input_values

```{autodoc2-docstring} src.utils.io.processing.find_single_input_values
```
````

````{py:function} find_two_input_values(data: typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]], current_path: str = '', input_key1: typing.Optional[str] = None, input_key2: typing.Union[str, int, float, None] = None) -> typing.List[typing.Tuple[str, typing.Union[int, float], typing.Union[int, float]]]
:canonical: src.utils.io.processing.find_two_input_values

```{autodoc2-docstring} src.utils.io.processing.find_two_input_values
```
````

````{py:function} process_pattern_files(root_directory: str, filename_pattern: str = 'log_*.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0, input_keys: typing.Tuple[typing.Optional[str], typing.Union[str, int, float, None]] = (None, None)) -> None
:canonical: src.utils.io.processing.process_pattern_files

```{autodoc2-docstring} src.utils.io.processing.process_pattern_files
```
````

````{py:function} process_file(file_path: str, output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0, input_keys: typing.Tuple[typing.Optional[str], typing.Union[str, int, float, None]] = (None, None)) -> bool
:canonical: src.utils.io.processing.process_file

```{autodoc2-docstring} src.utils.io.processing.process_file
```
````

````{py:function} process_pattern_files_statistics(root_directory: str, filename_pattern: str = 'log_*.json', output_filename: str = 'output.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.List[typing.Union[int, float]]], typing.Union[int, float]]] = None) -> typing.Optional[int]
:canonical: src.utils.io.processing.process_pattern_files_statistics

```{autodoc2-docstring} src.utils.io.processing.process_pattern_files_statistics
```
````

````{py:function} process_file_statistics(file_path: str, output_filename: str = 'output.json', output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.List[typing.Union[int, float]]], typing.Union[int, float]]] = None) -> bool
:canonical: src.utils.io.processing.process_file_statistics

```{autodoc2-docstring} src.utils.io.processing.process_file_statistics
```
````
