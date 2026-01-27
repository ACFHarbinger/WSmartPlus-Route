# {py:mod}`src.utils.functions.function`

```{py:module} src.utils.functions.function
```

```{autodoc2-docstring} src.utils.functions.function
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_inner_model <src.utils.functions.function.get_inner_model>`
  - ```{autodoc2-docstring} src.utils.functions.function.get_inner_model
    :summary:
    ```
* - {py:obj}`load_problem <src.utils.functions.function.load_problem>`
  - ```{autodoc2-docstring} src.utils.functions.function.load_problem
    :summary:
    ```
* - {py:obj}`torch_load_cpu <src.utils.functions.function.torch_load_cpu>`
  - ```{autodoc2-docstring} src.utils.functions.function.torch_load_cpu
    :summary:
    ```
* - {py:obj}`load_data <src.utils.functions.function.load_data>`
  - ```{autodoc2-docstring} src.utils.functions.function.load_data
    :summary:
    ```
* - {py:obj}`move_to <src.utils.functions.function.move_to>`
  - ```{autodoc2-docstring} src.utils.functions.function.move_to
    :summary:
    ```
* - {py:obj}`_load_model_file <src.utils.functions.function._load_model_file>`
  - ```{autodoc2-docstring} src.utils.functions.function._load_model_file
    :summary:
    ```
* - {py:obj}`load_args <src.utils.functions.function.load_args>`
  - ```{autodoc2-docstring} src.utils.functions.function.load_args
    :summary:
    ```
* - {py:obj}`load_model <src.utils.functions.function.load_model>`
  - ```{autodoc2-docstring} src.utils.functions.function.load_model
    :summary:
    ```
* - {py:obj}`parse_softmax_temperature <src.utils.functions.function.parse_softmax_temperature>`
  - ```{autodoc2-docstring} src.utils.functions.function.parse_softmax_temperature
    :summary:
    ```
* - {py:obj}`run_all_in_pool <src.utils.functions.function.run_all_in_pool>`
  - ```{autodoc2-docstring} src.utils.functions.function.run_all_in_pool
    :summary:
    ```
* - {py:obj}`get_path_until_string <src.utils.functions.function.get_path_until_string>`
  - ```{autodoc2-docstring} src.utils.functions.function.get_path_until_string
    :summary:
    ```
* - {py:obj}`compute_in_batches <src.utils.functions.function.compute_in_batches>`
  - ```{autodoc2-docstring} src.utils.functions.function.compute_in_batches
    :summary:
    ```
* - {py:obj}`add_attention_hooks <src.utils.functions.function.add_attention_hooks>`
  - ```{autodoc2-docstring} src.utils.functions.function.add_attention_hooks
    :summary:
    ```
* - {py:obj}`do_batch_rep <src.utils.functions.function.do_batch_rep>`
  - ```{autodoc2-docstring} src.utils.functions.function.do_batch_rep
    :summary:
    ```
* - {py:obj}`sample_many <src.utils.functions.function.sample_many>`
  - ```{autodoc2-docstring} src.utils.functions.function.sample_many
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <src.utils.functions.function.T>`
  - ```{autodoc2-docstring} src.utils.functions.function.T
    :summary:
    ```
````

### API

````{py:data} T
:canonical: src.utils.functions.function.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} src.utils.functions.function.T
```

````

````{py:function} get_inner_model(model: torch.nn.Module) -> torch.nn.Module
:canonical: src.utils.functions.function.get_inner_model

```{autodoc2-docstring} src.utils.functions.function.get_inner_model
```
````

````{py:function} load_problem(name: str) -> typing.Type[typing.Any]
:canonical: src.utils.functions.function.load_problem

```{autodoc2-docstring} src.utils.functions.function.load_problem
```
````

````{py:function} torch_load_cpu(load_path: str) -> typing.Any
:canonical: src.utils.functions.function.torch_load_cpu

```{autodoc2-docstring} src.utils.functions.function.torch_load_cpu
```
````

````{py:function} load_data(load_path: typing.Optional[str], resume: typing.Optional[str]) -> typing.Any
:canonical: src.utils.functions.function.load_data

```{autodoc2-docstring} src.utils.functions.function.load_data
```
````

````{py:function} move_to(var: src.utils.functions.function.T, device: torch.device, non_blocking: bool = False) -> src.utils.functions.function.T
:canonical: src.utils.functions.function.move_to

```{autodoc2-docstring} src.utils.functions.function.move_to
```
````

````{py:function} _load_model_file(load_path: str, model: torch.nn.Module) -> typing.Tuple[torch.nn.Module, typing.Optional[typing.Dict[str, typing.Any]]]
:canonical: src.utils.functions.function._load_model_file

```{autodoc2-docstring} src.utils.functions.function._load_model_file
```
````

````{py:function} load_args(filename: str) -> typing.Dict[str, typing.Any]
:canonical: src.utils.functions.function.load_args

```{autodoc2-docstring} src.utils.functions.function.load_args
```
````

````{py:function} load_model(path: str, epoch: typing.Optional[int] = None) -> typing.Tuple[torch.nn.Module, typing.Dict[str, typing.Any]]
:canonical: src.utils.functions.function.load_model

```{autodoc2-docstring} src.utils.functions.function.load_model
```
````

````{py:function} parse_softmax_temperature(raw_temp: typing.Union[str, float]) -> float
:canonical: src.utils.functions.function.parse_softmax_temperature

```{autodoc2-docstring} src.utils.functions.function.parse_softmax_temperature
```
````

````{py:function} run_all_in_pool(func: typing.Callable[..., typing.Any], directory: str, dataset: typing.List[typing.Any], opts: typing.Any, use_multiprocessing: bool = True) -> typing.Tuple[typing.List[typing.Any], int]
:canonical: src.utils.functions.function.run_all_in_pool

```{autodoc2-docstring} src.utils.functions.function.run_all_in_pool
```
````

````{py:function} get_path_until_string(path: str, end_str: str) -> typing.Optional[str]
:canonical: src.utils.functions.function.get_path_until_string

```{autodoc2-docstring} src.utils.functions.function.get_path_until_string
```
````

````{py:function} compute_in_batches(f: typing.Callable[..., typing.Any], calc_batch_size: int, *args: torch.Tensor, n: typing.Optional[int] = None) -> typing.Any
:canonical: src.utils.functions.function.compute_in_batches

```{autodoc2-docstring} src.utils.functions.function.compute_in_batches
```
````

````{py:function} add_attention_hooks(model_module: torch.nn.Module) -> typing.Dict[str, typing.Any]
:canonical: src.utils.functions.function.add_attention_hooks

```{autodoc2-docstring} src.utils.functions.function.add_attention_hooks
```
````

````{py:function} do_batch_rep(v: typing.Any, n: int) -> typing.Any
:canonical: src.utils.functions.function.do_batch_rep

```{autodoc2-docstring} src.utils.functions.function.do_batch_rep
```
````

````{py:function} sample_many(inner_func: typing.Callable[..., typing.Any], get_cost_func: typing.Callable[..., typing.Any], input: typing.Any, batch_rep: int = 1, iter_rep: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.utils.functions.function.sample_many

```{autodoc2-docstring} src.utils.functions.function.sample_many
```
````
