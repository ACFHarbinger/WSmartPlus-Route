# {py:mod}`src.utils.io.value_processing`

```{py:module} src.utils.io.value_processing
```

```{autodoc2-docstring} src.utils.io.value_processing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`find_single_input_values <src.utils.io.value_processing.find_single_input_values>`
  - ```{autodoc2-docstring} src.utils.io.value_processing.find_single_input_values
    :summary:
    ```
* - {py:obj}`_extract_two_vals <src.utils.io.value_processing._extract_two_vals>`
  - ```{autodoc2-docstring} src.utils.io.value_processing._extract_two_vals
    :summary:
    ```
* - {py:obj}`_process_val_pair <src.utils.io.value_processing._process_val_pair>`
  - ```{autodoc2-docstring} src.utils.io.value_processing._process_val_pair
    :summary:
    ```
* - {py:obj}`find_two_input_values <src.utils.io.value_processing.find_two_input_values>`
  - ```{autodoc2-docstring} src.utils.io.value_processing.find_two_input_values
    :summary:
    ```
````

### API

````{py:function} find_single_input_values(data: typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]], current_path: str = '', output_key: str = 'km') -> typing.List[typing.Tuple[str, typing.Any]]
:canonical: src.utils.io.value_processing.find_single_input_values

```{autodoc2-docstring} src.utils.io.value_processing.find_single_input_values
```
````

````{py:function} _extract_two_vals(data: typing.Any, k1: str, k2_or_val: typing.Any) -> typing.Tuple[typing.Any, typing.Any, bool]
:canonical: src.utils.io.value_processing._extract_two_vals

```{autodoc2-docstring} src.utils.io.value_processing._extract_two_vals
```
````

````{py:function} _process_val_pair(path: str, v1: typing.Any, v2: typing.Any, results: typing.List[typing.Tuple[str, typing.Any, typing.Any]])
:canonical: src.utils.io.value_processing._process_val_pair

```{autodoc2-docstring} src.utils.io.value_processing._process_val_pair
```
````

````{py:function} find_two_input_values(data: typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]], current_path: str = '', input_key1: typing.Optional[str] = None, input_key2: typing.Union[str, int, float, None] = None) -> typing.List[typing.Tuple[str, typing.Any, typing.Any]]
:canonical: src.utils.io.value_processing.find_two_input_values

```{autodoc2-docstring} src.utils.io.value_processing.find_two_input_values
```
````
