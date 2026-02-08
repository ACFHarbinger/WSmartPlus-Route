# {py:mod}`src.policies.operators.repair.regret`

```{py:module} src.policies.operators.repair.regret
```

```{autodoc2-docstring} src.policies.operators.repair.regret
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`regret_2_insertion <src.policies.operators.repair.regret.regret_2_insertion>`
  - ```{autodoc2-docstring} src.policies.operators.repair.regret.regret_2_insertion
    :summary:
    ```
* - {py:obj}`regret_k_insertion <src.policies.operators.repair.regret.regret_k_insertion>`
  - ```{autodoc2-docstring} src.policies.operators.repair.regret.regret_k_insertion
    :summary:
    ```
````

### API

````{py:function} regret_2_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float) -> typing.List[typing.List[int]]
:canonical: src.policies.operators.repair.regret.regret_2_insertion

```{autodoc2-docstring} src.policies.operators.repair.regret.regret_2_insertion
```
````

````{py:function} regret_k_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, k: int = 3) -> typing.List[typing.List[int]]
:canonical: src.policies.operators.repair.regret.regret_k_insertion

```{autodoc2-docstring} src.policies.operators.repair.regret.regret_k_insertion
```
````
