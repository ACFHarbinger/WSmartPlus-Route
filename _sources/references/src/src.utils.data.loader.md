# {py:mod}`src.utils.data.loader`

```{py:module} src.utils.data.loader
```

```{autodoc2-docstring} src.utils.data.loader
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`collate_fn <src.utils.data.loader.collate_fn>`
  - ```{autodoc2-docstring} src.utils.data.loader.collate_fn
    :summary:
    ```
* - {py:obj}`load_focus_coords <src.utils.data.loader.load_focus_coords>`
  - ```{autodoc2-docstring} src.utils.data.loader.load_focus_coords
    :summary:
    ```
* - {py:obj}`check_extension <src.utils.data.loader.check_extension>`
  - ```{autodoc2-docstring} src.utils.data.loader.check_extension
    :summary:
    ```
* - {py:obj}`save_dataset <src.utils.data.loader.save_dataset>`
  - ```{autodoc2-docstring} src.utils.data.loader.save_dataset
    :summary:
    ```
* - {py:obj}`save_td_dataset <src.utils.data.loader.save_td_dataset>`
  - ```{autodoc2-docstring} src.utils.data.loader.save_td_dataset
    :summary:
    ```
* - {py:obj}`load_td_dataset <src.utils.data.loader.load_td_dataset>`
  - ```{autodoc2-docstring} src.utils.data.loader.load_td_dataset
    :summary:
    ```
* - {py:obj}`load_dataset <src.utils.data.loader.load_dataset>`
  - ```{autodoc2-docstring} src.utils.data.loader.load_dataset
    :summary:
    ```
````

### API

````{py:function} collate_fn(batch: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]
:canonical: src.utils.data.loader.collate_fn

```{autodoc2-docstring} src.utils.data.loader.collate_fn
```
````

````{py:function} load_focus_coords(graph_size: int, method: typing.Optional[str], area: str, waste_type: str, focus_graph: str, focus_size: int = 1) -> typing.Tuple[numpy.ndarray, numpy.ndarray, typing.Optional[numpy.ndarray], typing.List[int]]
:canonical: src.utils.data.loader.load_focus_coords

```{autodoc2-docstring} src.utils.data.loader.load_focus_coords
```
````

````{py:function} check_extension(filename: str, extension: str = '.pkl') -> str
:canonical: src.utils.data.loader.check_extension

```{autodoc2-docstring} src.utils.data.loader.check_extension
```
````

````{py:function} save_dataset(dataset: typing.Any, filename: str) -> None
:canonical: src.utils.data.loader.save_dataset

```{autodoc2-docstring} src.utils.data.loader.save_dataset
```
````

````{py:function} save_td_dataset(td: tensordict.TensorDict, filename: str) -> None
:canonical: src.utils.data.loader.save_td_dataset

```{autodoc2-docstring} src.utils.data.loader.save_td_dataset
```
````

````{py:function} load_td_dataset(filename: str, device: str = 'cpu') -> tensordict.TensorDict
:canonical: src.utils.data.loader.load_td_dataset

```{autodoc2-docstring} src.utils.data.loader.load_td_dataset
```
````

````{py:function} load_dataset(filename: str) -> typing.Any
:canonical: src.utils.data.loader.load_dataset

```{autodoc2-docstring} src.utils.data.loader.load_dataset
```
````
