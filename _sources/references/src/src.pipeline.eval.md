# {py:mod}`src.pipeline.eval`

```{py:module} src.pipeline.eval
```

```{autodoc2-docstring} src.pipeline.eval
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_best <src.pipeline.eval.get_best>`
  - ```{autodoc2-docstring} src.pipeline.eval.get_best
    :summary:
    ```
* - {py:obj}`eval_dataset_mp <src.pipeline.eval.eval_dataset_mp>`
  - ```{autodoc2-docstring} src.pipeline.eval.eval_dataset_mp
    :summary:
    ```
* - {py:obj}`eval_dataset <src.pipeline.eval.eval_dataset>`
  - ```{autodoc2-docstring} src.pipeline.eval.eval_dataset
    :summary:
    ```
* - {py:obj}`_eval_dataset <src.pipeline.eval._eval_dataset>`
  - ```{autodoc2-docstring} src.pipeline.eval._eval_dataset
    :summary:
    ```
* - {py:obj}`run_evaluate_model <src.pipeline.eval.run_evaluate_model>`
  - ```{autodoc2-docstring} src.pipeline.eval.run_evaluate_model
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`mp <src.pipeline.eval.mp>`
  - ```{autodoc2-docstring} src.pipeline.eval.mp
    :summary:
    ```
````

### API

````{py:data} mp
:canonical: src.pipeline.eval.mp
:value: >
   'get_context(...)'

```{autodoc2-docstring} src.pipeline.eval.mp
```

````

````{py:function} get_best(sequences: numpy.ndarray, cost: numpy.ndarray, ids: typing.Optional[numpy.ndarray] = None, batch_size: typing.Optional[int] = None) -> typing.Tuple[typing.List[typing.Optional[numpy.ndarray]], typing.List[float]]
:canonical: src.pipeline.eval.get_best

```{autodoc2-docstring} src.pipeline.eval.get_best
```
````

````{py:function} eval_dataset_mp(args: typing.Tuple[str, int, float, typing.Dict[str, typing.Any], int, int]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.eval.eval_dataset_mp

```{autodoc2-docstring} src.pipeline.eval.eval_dataset_mp
```
````

````{py:function} eval_dataset(dataset_path: str, width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any], method: typing.Optional[str] = None) -> typing.Tuple[typing.List[float], typing.List[typing.Optional[typing.List[int]]], typing.List[float]]
:canonical: src.pipeline.eval.eval_dataset

```{autodoc2-docstring} src.pipeline.eval.eval_dataset
```
````

````{py:function} _eval_dataset(model: torch.nn.Module, dataset: torch.utils.data.Dataset, width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any], device: torch.device) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.eval._eval_dataset

```{autodoc2-docstring} src.pipeline.eval._eval_dataset
```
````

````{py:function} run_evaluate_model(opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.eval.run_evaluate_model

```{autodoc2-docstring} src.pipeline.eval.run_evaluate_model
```
````
