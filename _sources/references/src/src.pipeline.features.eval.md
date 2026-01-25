# {py:mod}`src.pipeline.features.eval`

```{py:module} src.pipeline.features.eval
```

```{autodoc2-docstring} src.pipeline.features.eval
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_best <src.pipeline.features.eval.get_best>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.get_best
    :summary:
    ```
* - {py:obj}`eval_dataset_mp <src.pipeline.features.eval.eval_dataset_mp>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.eval_dataset_mp
    :summary:
    ```
* - {py:obj}`eval_dataset <src.pipeline.features.eval.eval_dataset>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.eval_dataset
    :summary:
    ```
* - {py:obj}`_eval_dataset <src.pipeline.features.eval._eval_dataset>`
  - ```{autodoc2-docstring} src.pipeline.features.eval._eval_dataset
    :summary:
    ```
* - {py:obj}`run_evaluate_model <src.pipeline.features.eval.run_evaluate_model>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.run_evaluate_model
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`mp <src.pipeline.features.eval.mp>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.mp
    :summary:
    ```
````

### API

````{py:data} mp
:canonical: src.pipeline.features.eval.mp
:value: >
   'get_context(...)'

```{autodoc2-docstring} src.pipeline.features.eval.mp
```

````

````{py:function} get_best(sequences: numpy.ndarray, cost: numpy.ndarray, ids: typing.Optional[numpy.ndarray] = None, batch_size: typing.Optional[int] = None) -> typing.Tuple[typing.List[typing.Optional[numpy.ndarray]], typing.List[float]]
:canonical: src.pipeline.features.eval.get_best

```{autodoc2-docstring} src.pipeline.features.eval.get_best
```
````

````{py:function} eval_dataset_mp(args: typing.Tuple[str, int, float, typing.Dict[str, typing.Any], int, int]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.features.eval.eval_dataset_mp

```{autodoc2-docstring} src.pipeline.features.eval.eval_dataset_mp
```
````

````{py:function} eval_dataset(dataset_path: str, width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any], method: typing.Optional[str] = None) -> typing.Tuple[typing.List[float], typing.List[typing.Optional[typing.List[int]]], typing.List[float]]
:canonical: src.pipeline.features.eval.eval_dataset

```{autodoc2-docstring} src.pipeline.features.eval.eval_dataset
```
````

````{py:function} _eval_dataset(model: torch.nn.Module, dataset: torch.utils.data.Dataset, width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any], device: torch.device) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.features.eval._eval_dataset

```{autodoc2-docstring} src.pipeline.features.eval._eval_dataset
```
````

````{py:function} run_evaluate_model(opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.features.eval.run_evaluate_model

```{autodoc2-docstring} src.pipeline.features.eval.run_evaluate_model
```
````
