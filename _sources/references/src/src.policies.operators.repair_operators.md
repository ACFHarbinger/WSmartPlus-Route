# {py:mod}`src.policies.operators.repair_operators`

```{py:module} src.policies.operators.repair_operators
```

```{autodoc2-docstring} src.policies.operators.repair_operators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`greedy_insertion <src.policies.operators.repair_operators.greedy_insertion>`
  - ```{autodoc2-docstring} src.policies.operators.repair_operators.greedy_insertion
    :summary:
    ```
* - {py:obj}`regret_2_insertion <src.policies.operators.repair_operators.regret_2_insertion>`
  - ```{autodoc2-docstring} src.policies.operators.repair_operators.regret_2_insertion
    :summary:
    ```
* - {py:obj}`regret_k_insertion <src.policies.operators.repair_operators.regret_k_insertion>`
  - ```{autodoc2-docstring} src.policies.operators.repair_operators.regret_k_insertion
    :summary:
    ```
* - {py:obj}`greedy_insertion_with_blinks <src.policies.operators.repair_operators.greedy_insertion_with_blinks>`
  - ```{autodoc2-docstring} src.policies.operators.repair_operators.greedy_insertion_with_blinks
    :summary:
    ```
````

### API

````{py:function} greedy_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float) -> typing.List[typing.List[int]]
:canonical: src.policies.operators.repair_operators.greedy_insertion

```{autodoc2-docstring} src.policies.operators.repair_operators.greedy_insertion
```
````

````{py:function} regret_2_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float) -> typing.List[typing.List[int]]
:canonical: src.policies.operators.repair_operators.regret_2_insertion

```{autodoc2-docstring} src.policies.operators.repair_operators.regret_2_insertion
```
````

````{py:function} regret_k_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, k: int = 3) -> typing.List[typing.List[int]]
:canonical: src.policies.operators.repair_operators.regret_k_insertion

```{autodoc2-docstring} src.policies.operators.repair_operators.regret_k_insertion
```
````

````{py:function} greedy_insertion_with_blinks(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, blink_rate: float = 0.2) -> typing.List[typing.List[int]]
:canonical: src.policies.operators.repair_operators.greedy_insertion_with_blinks

```{autodoc2-docstring} src.policies.operators.repair_operators.greedy_insertion_with_blinks
```
````
