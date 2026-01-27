# {py:mod}`src.utils.data.data_utils`

```{py:module} src.utils.data.data_utils
```

```{autodoc2-docstring} src.utils.data.data_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`check_extension <src.utils.data.data_utils.check_extension>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.check_extension
    :summary:
    ```
* - {py:obj}`save_dataset <src.utils.data.data_utils.save_dataset>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.save_dataset
    :summary:
    ```
* - {py:obj}`save_td_dataset <src.utils.data.data_utils.save_td_dataset>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.save_td_dataset
    :summary:
    ```
* - {py:obj}`load_td_dataset <src.utils.data.data_utils.load_td_dataset>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.load_td_dataset
    :summary:
    ```
* - {py:obj}`load_dataset <src.utils.data.data_utils.load_dataset>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.load_dataset
    :summary:
    ```
* - {py:obj}`collate_fn <src.utils.data.data_utils.collate_fn>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.collate_fn
    :summary:
    ```
* - {py:obj}`load_focus_coords <src.utils.data.data_utils.load_focus_coords>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.load_focus_coords
    :summary:
    ```
* - {py:obj}`_get_fill_gamma <src.utils.data.data_utils._get_fill_gamma>`
  - ```{autodoc2-docstring} src.utils.data.data_utils._get_fill_gamma
    :summary:
    ```
* - {py:obj}`generate_waste_prize <src.utils.data.data_utils.generate_waste_prize>`
  - ```{autodoc2-docstring} src.utils.data.data_utils.generate_waste_prize
    :summary:
    ```
````

### API

````{py:function} check_extension(filename: str, extension: str = '.pkl') -> str
:canonical: src.utils.data.data_utils.check_extension

```{autodoc2-docstring} src.utils.data.data_utils.check_extension
```
````

````{py:function} save_dataset(dataset: typing.Any, filename: str) -> None
:canonical: src.utils.data.data_utils.save_dataset

```{autodoc2-docstring} src.utils.data.data_utils.save_dataset
```
````

````{py:function} save_td_dataset(td: tensordict.TensorDict, filename: str) -> None
:canonical: src.utils.data.data_utils.save_td_dataset

```{autodoc2-docstring} src.utils.data.data_utils.save_td_dataset
```
````

````{py:function} load_td_dataset(filename: str, device: str = 'cpu') -> tensordict.TensorDict
:canonical: src.utils.data.data_utils.load_td_dataset

```{autodoc2-docstring} src.utils.data.data_utils.load_td_dataset
```
````

````{py:function} load_dataset(filename: str) -> typing.Any
:canonical: src.utils.data.data_utils.load_dataset

```{autodoc2-docstring} src.utils.data.data_utils.load_dataset
```
````

````{py:function} collate_fn(batch: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]
:canonical: src.utils.data.data_utils.collate_fn

```{autodoc2-docstring} src.utils.data.data_utils.collate_fn
```
````

````{py:function} load_focus_coords(graph_size: int, method: typing.Optional[str], area: str, waste_type: str, focus_graph: str, focus_size: int = 1) -> typing.Tuple[numpy.ndarray, numpy.ndarray, typing.Optional[numpy.ndarray], typing.List[int]]
:canonical: src.utils.data.data_utils.load_focus_coords

```{autodoc2-docstring} src.utils.data.data_utils.load_focus_coords
```
````

````{py:function} _get_fill_gamma(dataset_size: int, problem_size: int, gamma_option: int) -> numpy.ndarray
:canonical: src.utils.data.data_utils._get_fill_gamma

```{autodoc2-docstring} src.utils.data.data_utils._get_fill_gamma
```
````

````{py:function} generate_waste_prize(problem_size: int, distribution: str, graph: typing.Tuple[typing.Any, typing.Any], dataset_size: int = 1, bins: typing.Optional[typing.Any] = None) -> typing.Union[numpy.ndarray, torch.Tensor]
:canonical: src.utils.data.data_utils.generate_waste_prize

```{autodoc2-docstring} src.utils.data.data_utils.generate_waste_prize
```
````
