# {py:mod}`src.policies.other.operators.perturbation.branch_bound`

```{py:module} src.policies.other.operators.perturbation.branch_bound
```

```{autodoc2-docstring} src.policies.other.operators.perturbation.branch_bound
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`bb_perturbation <src.policies.other.operators.perturbation.branch_bound.bb_perturbation>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.branch_bound.bb_perturbation
    :summary:
    ```
* - {py:obj}`bb_profit_perturbation <src.policies.other.operators.perturbation.branch_bound.bb_profit_perturbation>`
  - ```{autodoc2-docstring} src.policies.other.operators.perturbation.branch_bound.bb_profit_perturbation
    :summary:
    ```
````

### API

````{py:function} bb_perturbation(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, n_remove: int, destroy_discrepancy: int = 1, repair_discrepancy: int = 2, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_repair_pool: bool = False, rng: typing.Optional[random.Random] = None, noise: float = 0.0, return_removed: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.perturbation.branch_bound.bb_perturbation

```{autodoc2-docstring} src.policies.other.operators.perturbation.branch_bound.bb_perturbation
```
````

````{py:function} bb_profit_perturbation(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, n_remove: int, destroy_discrepancy: int = 1, repair_discrepancy: int = 2, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_repair_pool: bool = False, rng: typing.Optional[random.Random] = None, noise: float = 0.0, seed_hurdle_factor: float = 0.5, return_removed: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.perturbation.branch_bound.bb_profit_perturbation

```{autodoc2-docstring} src.policies.other.operators.perturbation.branch_bound.bb_profit_perturbation
```
````
