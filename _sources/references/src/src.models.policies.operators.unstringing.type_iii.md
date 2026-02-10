# {py:mod}`src.models.policies.operators.unstringing.type_iii`

```{py:module} src.models.policies.operators.unstringing.type_iii
```

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_type_iii_unstringing <src.models.policies.operators.unstringing.type_iii.vectorized_type_iii_unstringing>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii.vectorized_type_iii_unstringing
    :summary:
    ```
* - {py:obj}`_find_best_type_iii_move <src.models.policies.operators.unstringing.type_iii._find_best_type_iii_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii._find_best_type_iii_move
    :summary:
    ```
* - {py:obj}`_evaluate_type_iii_move <src.models.policies.operators.unstringing.type_iii._evaluate_type_iii_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii._evaluate_type_iii_move
    :summary:
    ```
* - {py:obj}`_apply_type_iii_move <src.models.policies.operators.unstringing.type_iii._apply_type_iii_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii._apply_type_iii_move
    :summary:
    ```
````

### API

````{py:function} vectorized_type_iii_unstringing(tours: torch.Tensor, distance_matrix: torch.Tensor, max_iterations: int = 50, sample_size: int = 50) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_iii.vectorized_type_iii_unstringing

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii.vectorized_type_iii_unstringing
```
````

````{py:function} _find_best_type_iii_move(tour, dist, valid_indices, sample_size, device)
:canonical: src.models.policies.operators.unstringing.type_iii._find_best_type_iii_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii._find_best_type_iii_move
```
````

````{py:function} _evaluate_type_iii_move(tour, dist, i, k, j, l, N)
:canonical: src.models.policies.operators.unstringing.type_iii._evaluate_type_iii_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii._evaluate_type_iii_move
```
````

````{py:function} _apply_type_iii_move(tour: torch.Tensor, i: int, k: int, j: int, l: int) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_iii._apply_type_iii_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iii._apply_type_iii_move
```
````
