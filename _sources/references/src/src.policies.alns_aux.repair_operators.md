# {py:mod}`src.policies.alns_aux.repair_operators`

```{py:module} src.policies.alns_aux.repair_operators
```

```{autodoc2-docstring} src.policies.alns_aux.repair_operators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`greedy_insertion <src.policies.alns_aux.repair_operators.greedy_insertion>`
  - ```{autodoc2-docstring} src.policies.alns_aux.repair_operators.greedy_insertion
    :summary:
    ```
* - {py:obj}`regret_2_insertion <src.policies.alns_aux.repair_operators.regret_2_insertion>`
  - ```{autodoc2-docstring} src.policies.alns_aux.repair_operators.regret_2_insertion
    :summary:
    ```
````

### API

````{py:function} greedy_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float) -> typing.List[typing.List[int]]
:canonical: src.policies.alns_aux.repair_operators.greedy_insertion

```{autodoc2-docstring} src.policies.alns_aux.repair_operators.greedy_insertion
```
````

````{py:function} regret_2_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float) -> typing.List[typing.List[int]]
:canonical: src.policies.alns_aux.repair_operators.regret_2_insertion

```{autodoc2-docstring} src.policies.alns_aux.repair_operators.regret_2_insertion
```
````
