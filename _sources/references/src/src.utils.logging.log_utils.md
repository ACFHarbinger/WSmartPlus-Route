# {py:mod}`src.utils.logging.log_utils`

```{py:module} src.utils.logging.log_utils
```

```{autodoc2-docstring} src.utils.logging.log_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup_system_logger <src.utils.logging.log_utils.setup_system_logger>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.setup_system_logger
    :summary:
    ```
* - {py:obj}`log_values <src.utils.logging.log_utils.log_values>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.log_values
    :summary:
    ```
* - {py:obj}`log_epoch <src.utils.logging.log_utils.log_epoch>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.log_epoch
    :summary:
    ```
* - {py:obj}`get_loss_stats <src.utils.logging.log_utils.get_loss_stats>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.get_loss_stats
    :summary:
    ```
* - {py:obj}`log_training <src.utils.logging.log_utils.log_training>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.log_training
    :summary:
    ```
* - {py:obj}`_sort_log <src.utils.logging.log_utils._sort_log>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils._sort_log
    :summary:
    ```
* - {py:obj}`sort_log <src.utils.logging.log_utils.sort_log>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.sort_log
    :summary:
    ```
* - {py:obj}`_convert_numpy <src.utils.logging.log_utils._convert_numpy>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils._convert_numpy
    :summary:
    ```
* - {py:obj}`log_to_json <src.utils.logging.log_utils.log_to_json>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.log_to_json
    :summary:
    ```
* - {py:obj}`log_to_json2 <src.utils.logging.log_utils.log_to_json2>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.log_to_json2
    :summary:
    ```
* - {py:obj}`log_to_pickle <src.utils.logging.log_utils.log_to_pickle>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.log_to_pickle
    :summary:
    ```
* - {py:obj}`update_log <src.utils.logging.log_utils.update_log>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.update_log
    :summary:
    ```
* - {py:obj}`load_log_dict <src.utils.logging.log_utils.load_log_dict>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.load_log_dict
    :summary:
    ```
* - {py:obj}`log_plot <src.utils.logging.log_utils.log_plot>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.log_plot
    :summary:
    ```
* - {py:obj}`output_stats <src.utils.logging.log_utils.output_stats>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.output_stats
    :summary:
    ```
* - {py:obj}`runs_per_policy <src.utils.logging.log_utils.runs_per_policy>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.runs_per_policy
    :summary:
    ```
* - {py:obj}`send_daily_output_to_gui <src.utils.logging.log_utils.send_daily_output_to_gui>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.send_daily_output_to_gui
    :summary:
    ```
* - {py:obj}`send_final_output_to_gui <src.utils.logging.log_utils.send_final_output_to_gui>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.send_final_output_to_gui
    :summary:
    ```
* - {py:obj}`final_simulation_summary <src.utils.logging.log_utils.final_simulation_summary>`
  - ```{autodoc2-docstring} src.utils.logging.log_utils.final_simulation_summary
    :summary:
    ```
````

### API

````{py:function} setup_system_logger(log_path: str = 'logs/system.log', level: str = 'INFO') -> typing.Any
:canonical: src.utils.logging.log_utils.setup_system_logger

```{autodoc2-docstring} src.utils.logging.log_utils.setup_system_logger
```
````

````{py:function} log_values(cost: torch.Tensor, grad_norms: typing.Tuple[torch.Tensor, ...], epoch: int, batch_id: int, step: int, l_dict: typing.Dict[str, torch.Tensor], tb_logger: typing.Any, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.logging.log_utils.log_values

```{autodoc2-docstring} src.utils.logging.log_utils.log_values
```
````

````{py:function} log_epoch(x_tup: typing.Tuple[str, int], loss_keys: typing.List[str], epoch_loss: typing.Dict[str, typing.List[torch.Tensor]], opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.logging.log_utils.log_epoch

```{autodoc2-docstring} src.utils.logging.log_utils.log_epoch
```
````

````{py:function} get_loss_stats(epoch_loss: typing.Dict[str, typing.List[torch.Tensor]]) -> typing.List[float]
:canonical: src.utils.logging.log_utils.get_loss_stats

```{autodoc2-docstring} src.utils.logging.log_utils.get_loss_stats
```
````

````{py:function} log_training(loss_keys: typing.List[str], table_df: pandas.DataFrame, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.logging.log_utils.log_training

```{autodoc2-docstring} src.utils.logging.log_utils.log_training
```
````

````{py:function} _sort_log(log: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]
:canonical: src.utils.logging.log_utils._sort_log

```{autodoc2-docstring} src.utils.logging.log_utils._sort_log
```
````

````{py:function} sort_log(logfile_path: str, lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.utils.logging.log_utils.sort_log

```{autodoc2-docstring} src.utils.logging.log_utils.sort_log
```
````

````{py:function} _convert_numpy(obj: typing.Any) -> typing.Any
:canonical: src.utils.logging.log_utils._convert_numpy

```{autodoc2-docstring} src.utils.logging.log_utils._convert_numpy
```
````

````{py:function} log_to_json(json_path: str, keys: typing.List[str], dit: typing.Dict[str, typing.Any], sort_log: bool = True, sample_id: typing.Optional[int] = None, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.logging.log_utils.log_to_json

```{autodoc2-docstring} src.utils.logging.log_utils.log_to_json
```
````

````{py:function} log_to_json2(json_path: str, keys: typing.List[str], dit: typing.Dict[str, typing.Any], sort_log: bool = True, sample_id: typing.Optional[int] = None, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.logging.log_utils.log_to_json2

```{autodoc2-docstring} src.utils.logging.log_utils.log_to_json2
```
````

````{py:function} log_to_pickle(pickle_path: str, log: typing.Any, lock: typing.Optional[threading.Lock] = None, dw_func: typing.Optional[typing.Callable[[str], None]] = None) -> None
:canonical: src.utils.logging.log_utils.log_to_pickle

```{autodoc2-docstring} src.utils.logging.log_utils.log_to_pickle
```
````

````{py:function} update_log(json_path: str, new_output: typing.List[typing.Dict[str, typing.Any]], start_id: int, policies: typing.List[str], sort_log: bool = True, lock: typing.Optional[threading.Lock] = None) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Any]]
:canonical: src.utils.logging.log_utils.update_log

```{autodoc2-docstring} src.utils.logging.log_utils.update_log
```
````

````{py:function} load_log_dict(dir_paths: typing.List[str], nsamples: typing.List[int], show_incomplete: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.Dict[str, str]
:canonical: src.utils.logging.log_utils.load_log_dict

```{autodoc2-docstring} src.utils.logging.log_utils.load_log_dict
```
````

````{py:function} log_plot(visualize: bool = False, **kwargs: typing.Any) -> None
:canonical: src.utils.logging.log_utils.log_plot

```{autodoc2-docstring} src.utils.logging.log_utils.log_plot
```
````

````{py:function} output_stats(dir_path: str, nsamples: int, policies: typing.List[str], keys: typing.List[str], sort_log: bool = True, print_output: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.Tuple[typing.Dict[str, typing.Any], typing.Dict[str, typing.Any]]
:canonical: src.utils.logging.log_utils.output_stats

```{autodoc2-docstring} src.utils.logging.log_utils.output_stats
```
````

````{py:function} runs_per_policy(dir_paths: typing.List[str], nsamples: typing.List[int], policies: typing.List[str], print_output: bool = False, lock: typing.Optional[threading.Lock] = None) -> typing.List[typing.Dict[str, typing.List[int]]]
:canonical: src.utils.logging.log_utils.runs_per_policy

```{autodoc2-docstring} src.utils.logging.log_utils.runs_per_policy
```
````

````{py:function} send_daily_output_to_gui(daily_log: typing.Dict[str, typing.Any], policy: str, sample_idx: int, day: int, bins_c: typing.Sequence[float], collected: typing.Sequence[float], bins_c_after: typing.Sequence[float], log_path: str, tour: typing.Sequence[int], coordinates: typing.Union[pandas.DataFrame, typing.List[typing.Any]], distance_matrix: typing.Optional[numpy.ndarray] = None, lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.utils.logging.log_utils.send_daily_output_to_gui

```{autodoc2-docstring} src.utils.logging.log_utils.send_daily_output_to_gui
```
````

````{py:function} send_final_output_to_gui(log: typing.Dict[str, typing.Any], log_std: typing.Optional[typing.Dict[str, typing.Any]], n_samples: int, policies: typing.List[str], log_path: str, lock: typing.Optional[threading.Lock] = None) -> None
:canonical: src.utils.logging.log_utils.send_final_output_to_gui

```{autodoc2-docstring} src.utils.logging.log_utils.send_final_output_to_gui
```
````

````{py:function} final_simulation_summary(log: typing.Dict[str, typing.Any], policy: str, n_samples: int) -> None
:canonical: src.utils.logging.log_utils.final_simulation_summary

```{autodoc2-docstring} src.utils.logging.log_utils.final_simulation_summary
```
````
