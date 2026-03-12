# {py:mod}`src.policies.other.operators.destroy.string`

```{py:module} src.policies.other.operators.destroy.string
```

```{autodoc2-docstring} src.policies.other.operators.destroy.string
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`string_removal <src.policies.other.operators.destroy.string.string_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.string.string_removal
    :summary:
    ```
* - {py:obj}`_propagate_string_removal <src.policies.other.operators.destroy.string._propagate_string_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.string._propagate_string_removal
    :summary:
    ```
* - {py:obj}`_get_node_profits <src.policies.other.operators.destroy.string._get_node_profits>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.string._get_node_profits
    :summary:
    ```
* - {py:obj}`_select_string_seed <src.policies.other.operators.destroy.string._select_string_seed>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.string._select_string_seed
    :summary:
    ```
* - {py:obj}`_get_string_length <src.policies.other.operators.destroy.string._get_string_length>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.string._get_string_length
    :summary:
    ```
* - {py:obj}`string_profit_removal <src.policies.other.operators.destroy.string.string_profit_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.string.string_profit_removal
    :summary:
    ```
* - {py:obj}`_propagate_profit_string_removal <src.policies.other.operators.destroy.string._propagate_profit_string_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.string._propagate_profit_string_removal
    :summary:
    ```
````

### API

````{py:function} string_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, max_string_len: int = 4, avg_string_len: float = 3.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.string.string_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.string.string_removal
```
````

````{py:function} _propagate_string_removal(routes: typing.List[typing.List[int]], removed: typing.List[int], dist_matrix: numpy.ndarray, seed_nodes: typing.List[int], n_remove: int, max_string_len: int) -> None
:canonical: src.policies.other.operators.destroy.string._propagate_string_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.string._propagate_string_removal
```
````

````{py:function} _get_node_profits(routes: typing.List[typing.List[int]], wastes: typing.Dict[int, float], dist_matrix: numpy.ndarray, R: float, C: float) -> typing.Dict[int, float]
:canonical: src.policies.other.operators.destroy.string._get_node_profits

```{autodoc2-docstring} src.policies.other.operators.destroy.string._get_node_profits
```
````

````{py:function} _select_string_seed(routes: typing.List[typing.List[int]], low_profit_nodes: typing.List[int], removed: typing.List[int], rng: random.Random) -> typing.Tuple[typing.Optional[int], int, int]
:canonical: src.policies.other.operators.destroy.string._select_string_seed

```{autodoc2-docstring} src.policies.other.operators.destroy.string._select_string_seed
```
````

````{py:function} _get_string_length(max_string_len: int, avg_string_len: float, remaining_to_remove: int, route_len: int, rng: random.Random) -> int
:canonical: src.policies.other.operators.destroy.string._get_string_length

```{autodoc2-docstring} src.policies.other.operators.destroy.string._get_string_length
```
````

````{py:function} string_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, max_string_len: int = 4, avg_string_len: float = 3.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.string.string_profit_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.string.string_profit_removal
```
````

````{py:function} _propagate_profit_string_removal(routes: typing.List[typing.List[int]], removed: typing.List[int], dist_matrix: numpy.ndarray, seed_nodes: typing.List[int], n_remove: int, node_profits: typing.Dict[int, float]) -> None
:canonical: src.policies.other.operators.destroy.string._propagate_profit_string_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.string._propagate_profit_string_removal
```
````
