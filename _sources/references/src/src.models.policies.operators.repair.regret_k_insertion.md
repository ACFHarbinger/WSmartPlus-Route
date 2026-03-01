# {py:mod}`src.models.policies.operators.repair.regret_k_insertion`

```{py:module} src.models.policies.operators.repair.regret_k_insertion
```

```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_regret_k_insertion <src.models.policies.operators.repair.regret_k_insertion.vectorized_regret_k_insertion>`
  - ```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion.vectorized_regret_k_insertion
    :summary:
    ```
* - {py:obj}`_compute_insertion_costs <src.models.policies.operators.repair.regret_k_insertion._compute_insertion_costs>`
  - ```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion._compute_insertion_costs
    :summary:
    ```
* - {py:obj}`_compute_regrets <src.models.policies.operators.repair.regret_k_insertion._compute_regrets>`
  - ```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion._compute_regrets
    :summary:
    ```
* - {py:obj}`_apply_insertion <src.models.policies.operators.repair.regret_k_insertion._apply_insertion>`
  - ```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion._apply_insertion
    :summary:
    ```
````

### API

````{py:function} vectorized_regret_k_insertion(tours: torch.Tensor, removed_nodes: torch.Tensor, dist_matrix: torch.Tensor, k: int = 2) -> torch.Tensor
:canonical: src.models.policies.operators.repair.regret_k_insertion.vectorized_regret_k_insertion

```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion.vectorized_regret_k_insertion
```
````

````{py:function} _compute_insertion_costs(tours, removed_nodes, dist)
:canonical: src.models.policies.operators.repair.regret_k_insertion._compute_insertion_costs

```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion._compute_insertion_costs
```
````

````{py:function} _compute_regrets(costs, k, pending_mask)
:canonical: src.models.policies.operators.repair.regret_k_insertion._compute_regrets

```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion._compute_regrets
```
````

````{py:function} _apply_insertion(tours, node, pos)
:canonical: src.models.policies.operators.repair.regret_k_insertion._apply_insertion

```{autodoc2-docstring} src.models.policies.operators.repair.regret_k_insertion._apply_insertion
```
````
