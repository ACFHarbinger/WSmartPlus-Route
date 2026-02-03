# {py:mod}`src.utils.logging.modules.storage`

```{py:module} src.utils.logging.modules.storage
```

```{autodoc2-docstring} src.utils.logging.modules.storage
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup_system_logger <src.utils.logging.modules.storage.setup_system_logger>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage.setup_system_logger
    :summary:
    ```
* - {py:obj}`_convert_numpy <src.utils.logging.modules.storage._convert_numpy>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage._convert_numpy
    :summary:
    ```
* - {py:obj}`_sort_log <src.utils.logging.modules.storage._sort_log>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage._sort_log
    :summary:
    ```
* - {py:obj}`sort_log <src.utils.logging.modules.storage.sort_log>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage.sort_log
    :summary:
    ```
* - {py:obj}`log_to_json <src.utils.logging.modules.storage.log_to_json>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage.log_to_json
    :summary:
    ```
* - {py:obj}`log_to_json2 <src.utils.logging.modules.storage.log_to_json2>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage.log_to_json2
    :summary:
    ```
* - {py:obj}`log_to_pickle <src.utils.logging.modules.storage.log_to_pickle>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage.log_to_pickle
    :summary:
    ```
* - {py:obj}`update_log <src.utils.logging.modules.storage.update_log>`
  - ```{autodoc2-docstring} src.utils.logging.modules.storage.update_log
    :summary:
    ```
````

### API

````{py:function} setup_system_logger(log_path: str = 'logs/system.log', level: str = 'INFO') -> typing.Any
:canonical: src.utils.logging.modules.storage.setup_system_logger

```{autodoc2-docstring} src.utils.logging.modules.storage.setup_system_logger
```
````

````{py:function} _convert_numpy(obj: typing.Any) -> typing.Any
:canonical: src.utils.logging.modules.storage._convert_numpy

```{autodoc2-docstring} src.utils.logging.modules.storage._convert_numpy
```
````

````{py:function} _sort_log(log: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]
:canonical: src.utils.logging.modules.storage._sort_log

```{autodoc2-docstring} src.utils.logging.modules.storage._sort_log
```
````

````{py:function} sort_log(logfile_path: str, lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.utils.logging.modules.storage.sort_log

```{autodoc2-docstring} src.utils.logging.modules.storage.sort_log
```
````

````{py:function} log_to_json(json_path: str, keys: typing.List[str], dit: typing.Dict[str, typing.Any], sort_log_flag: bool = True, sample_id: typing.Optional[int] = None, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.logging.modules.storage.log_to_json

```{autodoc2-docstring} src.utils.logging.modules.storage.log_to_json
```
````

````{py:function} log_to_json2(json_path: str, keys: typing.List[str], dit: typing.Dict[str, typing.Any], sort_log_flag: bool = True, sample_id: typing.Optional[int] = None, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.logging.modules.storage.log_to_json2

```{autodoc2-docstring} src.utils.logging.modules.storage.log_to_json2
```
````

````{py:function} log_to_pickle(pickle_path: str, log: typing.Any, lock: typing.Optional[threading.Lock] = None, dw_func: typing.Optional[typing.Callable[[str], None]] = None) -> None
:canonical: src.utils.logging.modules.storage.log_to_pickle

```{autodoc2-docstring} src.utils.logging.modules.storage.log_to_pickle
```
````

````{py:function} update_log(json_path: str, new_output: typing.List[typing.Dict[str, typing.Any]], start_id: int, policies: typing.List[str], sort_log_flag: bool = True, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.logging.modules.storage.update_log

```{autodoc2-docstring} src.utils.logging.modules.storage.update_log
```
````
