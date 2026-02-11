# {py:mod}`src.utils.io.dict_processing`

```{py:module} src.utils.io.dict_processing
```

```{autodoc2-docstring} src.utils.io.dict_processing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`process_dict_of_dicts <src.utils.io.dict_processing.process_dict_of_dicts>`
  - ```{autodoc2-docstring} src.utils.io.dict_processing.process_dict_of_dicts
    :summary:
    ```
* - {py:obj}`process_list_of_dicts <src.utils.io.dict_processing.process_list_of_dicts>`
  - ```{autodoc2-docstring} src.utils.io.dict_processing.process_list_of_dicts
    :summary:
    ```
* - {py:obj}`process_dict_two_inputs <src.utils.io.dict_processing.process_dict_two_inputs>`
  - ```{autodoc2-docstring} src.utils.io.dict_processing.process_dict_two_inputs
    :summary:
    ```
* - {py:obj}`process_list_two_inputs <src.utils.io.dict_processing.process_list_two_inputs>`
  - ```{autodoc2-docstring} src.utils.io.dict_processing.process_list_two_inputs
    :summary:
    ```
````

### API

````{py:function} process_dict_of_dicts(data_dict: typing.Dict[str, typing.Any], output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0) -> bool
:canonical: src.utils.io.dict_processing.process_dict_of_dicts

```{autodoc2-docstring} src.utils.io.dict_processing.process_dict_of_dicts
```
````

````{py:function} process_list_of_dicts(data_list: typing.List[typing.Dict[str, typing.Any]], output_key: str = 'km', process_func: typing.Optional[typing.Callable[[typing.Any, typing.Any], typing.Any]] = None, update_val: typing.Union[int, float] = 0) -> bool
:canonical: src.utils.io.dict_processing.process_list_of_dicts

```{autodoc2-docstring} src.utils.io.dict_processing.process_list_of_dicts
```
````

````{py:function} process_dict_two_inputs(data_dict: typing.Union[typing.Dict[str, typing.Any], typing.Any], input_key1: str, input_key2_or_val: typing.Union[str, int, float, None], output_key: str, process_func: typing.Callable[[typing.Any, typing.Any], typing.Any]) -> bool
:canonical: src.utils.io.dict_processing.process_dict_two_inputs

```{autodoc2-docstring} src.utils.io.dict_processing.process_dict_two_inputs
```
````

````{py:function} process_list_two_inputs(data_list: typing.List[typing.Dict[str, typing.Any]], input_key1: str, input_key2_or_val: typing.Union[str, int, float], output_key: str, process_func: typing.Callable[[typing.Any, typing.Any], typing.Any]) -> bool
:canonical: src.utils.io.dict_processing.process_list_two_inputs

```{autodoc2-docstring} src.utils.io.dict_processing.process_list_two_inputs
```
````
