# {py:mod}`src.models.policies.operators.unstringing.type_ii`

```{py:module} src.models.policies.operators.unstringing.type_ii
```

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_type_ii_unstringing <src.models.policies.operators.unstringing.type_ii.vectorized_type_ii_unstringing>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii.vectorized_type_ii_unstringing
    :summary:
    ```
* - {py:obj}`_find_best_type_ii_move <src.models.policies.operators.unstringing.type_ii._find_best_type_ii_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii._find_best_type_ii_move
    :summary:
    ```
* - {py:obj}`_evaluate_type_ii_move <src.models.policies.operators.unstringing.type_ii._evaluate_type_ii_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii._evaluate_type_ii_move
    :summary:
    ```
* - {py:obj}`_apply_type_ii_move <src.models.policies.operators.unstringing.type_ii._apply_type_ii_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii._apply_type_ii_move
    :summary:
    ```
````

### API

````{py:function} vectorized_type_ii_unstringing(tours: torch.Tensor, distance_matrix: torch.Tensor, max_iterations: int = 50, sample_size: int = 100) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_ii.vectorized_type_ii_unstringing

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii.vectorized_type_ii_unstringing
```
````

````{py:function} _find_best_type_ii_move(tour, dist, valid_indices, sample_size, device)
:canonical: src.models.policies.operators.unstringing.type_ii._find_best_type_ii_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii._find_best_type_ii_move
```
````

````{py:function} _evaluate_type_ii_move(tour, dist, i, j, k, N)
:canonical: src.models.policies.operators.unstringing.type_ii._evaluate_type_ii_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii._evaluate_type_ii_move
```
````

````{py:function} _apply_type_ii_move(tour: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_ii._apply_type_ii_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_ii._apply_type_ii_move
```
````
