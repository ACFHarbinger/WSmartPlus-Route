# {py:mod}`src.models.policies.operators.repair.greedy_insertion`

```{py:module} src.models.policies.operators.repair.greedy_insertion
```

```{autodoc2-docstring} src.models.policies.operators.repair.greedy_insertion
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_greedy_insertion <src.models.policies.operators.repair.greedy_insertion.vectorized_greedy_insertion>`
  - ```{autodoc2-docstring} src.models.policies.operators.repair.greedy_insertion.vectorized_greedy_insertion
    :summary:
    ```
* - {py:obj}`_compute_greedy_insertion_costs <src.models.policies.operators.repair.greedy_insertion._compute_greedy_insertion_costs>`
  - ```{autodoc2-docstring} src.models.policies.operators.repair.greedy_insertion._compute_greedy_insertion_costs
    :summary:
    ```
* - {py:obj}`_apply_insertion <src.models.policies.operators.repair.greedy_insertion._apply_insertion>`
  - ```{autodoc2-docstring} src.models.policies.operators.repair.greedy_insertion._apply_insertion
    :summary:
    ```
````

### API

````{py:function} vectorized_greedy_insertion(tours: torch.Tensor, removed_nodes: torch.Tensor, dist_matrix: torch.Tensor, wastes: typing.Optional[torch.Tensor] = None, capacity: typing.Optional[float] = None) -> torch.Tensor
:canonical: src.models.policies.operators.repair.greedy_insertion.vectorized_greedy_insertion

```{autodoc2-docstring} src.models.policies.operators.repair.greedy_insertion.vectorized_greedy_insertion
```
````

````{py:function} _compute_greedy_insertion_costs(tours, node, dist)
:canonical: src.models.policies.operators.repair.greedy_insertion._compute_greedy_insertion_costs

```{autodoc2-docstring} src.models.policies.operators.repair.greedy_insertion._compute_greedy_insertion_costs
```
````

````{py:function} _apply_insertion(tours, node, pos)
:canonical: src.models.policies.operators.repair.greedy_insertion._apply_insertion

```{autodoc2-docstring} src.models.policies.operators.repair.greedy_insertion._apply_insertion
```
````
