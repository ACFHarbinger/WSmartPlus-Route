# {py:mod}`src.pipeline.features.eval.engine`

```{py:module} src.pipeline.features.eval.engine
```

```{autodoc2-docstring} src.pipeline.features.eval.engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_best <src.pipeline.features.eval.engine.get_best>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine.get_best
    :summary:
    ```
* - {py:obj}`eval_dataset_mp <src.pipeline.features.eval.engine.eval_dataset_mp>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine.eval_dataset_mp
    :summary:
    ```
* - {py:obj}`_eval_dataset <src.pipeline.features.eval.engine._eval_dataset>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine._eval_dataset
    :summary:
    ```
* - {py:obj}`eval_dataset <src.pipeline.features.eval.engine.eval_dataset>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine.eval_dataset
    :summary:
    ```
* - {py:obj}`_eval_multiprocessing <src.pipeline.features.eval.engine._eval_multiprocessing>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine._eval_multiprocessing
    :summary:
    ```
* - {py:obj}`_eval_singleprocess <src.pipeline.features.eval.engine._eval_singleprocess>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine._eval_singleprocess
    :summary:
    ```
* - {py:obj}`_get_eval_output_path <src.pipeline.features.eval.engine._get_eval_output_path>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine._get_eval_output_path
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`mp <src.pipeline.features.eval.engine.mp>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.engine.mp
    :summary:
    ```
````

### API

````{py:data} mp
:canonical: src.pipeline.features.eval.engine.mp
:value: >
   'get_context(...)'

```{autodoc2-docstring} src.pipeline.features.eval.engine.mp
```

````

````{py:function} get_best(sequences: numpy.ndarray, cost: numpy.ndarray, ids: typing.Optional[numpy.ndarray] = None, batch_size: typing.Optional[int] = None) -> typing.Tuple[typing.List[typing.Optional[numpy.ndarray]], typing.List[float]]
:canonical: src.pipeline.features.eval.engine.get_best

```{autodoc2-docstring} src.pipeline.features.eval.engine.get_best
```
````

````{py:function} eval_dataset_mp(args: typing.Tuple[str, int, float, typing.Dict[str, typing.Any], int, int]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.features.eval.engine.eval_dataset_mp

```{autodoc2-docstring} src.pipeline.features.eval.engine.eval_dataset_mp
```
````

````{py:function} _eval_dataset(model: torch.nn.Module, dataset: torch.utils.data.Dataset, beam_width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any], device: torch.device) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.features.eval.engine._eval_dataset

```{autodoc2-docstring} src.pipeline.features.eval.engine._eval_dataset
```
````

````{py:function} eval_dataset(dataset_path: str, beam_width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any], method: typing.Optional[str] = None) -> typing.Tuple[typing.List[float], typing.List[typing.Optional[typing.List[int]]], typing.List[float]]
:canonical: src.pipeline.features.eval.engine.eval_dataset

```{autodoc2-docstring} src.pipeline.features.eval.engine.eval_dataset
```
````

````{py:function} _eval_multiprocessing(dataset_path: str, beam_width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.features.eval.engine._eval_multiprocessing

```{autodoc2-docstring} src.pipeline.features.eval.engine._eval_multiprocessing
```
````

````{py:function} _eval_singleprocess(model: torch.nn.Module, dataset_path: str, beam_width: int, softmax_temp: float, opts: typing.Dict[str, typing.Any], use_cuda: bool) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.features.eval.engine._eval_singleprocess

```{autodoc2-docstring} src.pipeline.features.eval.engine._eval_singleprocess
```
````

````{py:function} _get_eval_output_path(model: torch.nn.Module, dataset_path: str, opts: typing.Dict[str, typing.Any], model_name: str, beam_width: int, softmax_temp: float, num_costs: int) -> str
:canonical: src.pipeline.features.eval.engine._get_eval_output_path

```{autodoc2-docstring} src.pipeline.features.eval.engine._get_eval_output_path
```
````
