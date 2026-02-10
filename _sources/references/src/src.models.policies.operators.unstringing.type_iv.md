# {py:mod}`src.models.policies.operators.unstringing.type_iv`

```{py:module} src.models.policies.operators.unstringing.type_iv
```

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_type_iv_unstringing <src.models.policies.operators.unstringing.type_iv.vectorized_type_iv_unstringing>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv.vectorized_type_iv_unstringing
    :summary:
    ```
* - {py:obj}`_find_best_type_iv_move <src.models.policies.operators.unstringing.type_iv._find_best_type_iv_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv._find_best_type_iv_move
    :summary:
    ```
* - {py:obj}`_evaluate_type_iv_move <src.models.policies.operators.unstringing.type_iv._evaluate_type_iv_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv._evaluate_type_iv_move
    :summary:
    ```
* - {py:obj}`_apply_type_iv_move <src.models.policies.operators.unstringing.type_iv._apply_type_iv_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv._apply_type_iv_move
    :summary:
    ```
````

### API

````{py:function} vectorized_type_iv_unstringing(tours: torch.Tensor, distance_matrix: torch.Tensor, max_iterations: int = 50, sample_size: int = 30) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_iv.vectorized_type_iv_unstringing

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv.vectorized_type_iv_unstringing
```
````

````{py:function} _find_best_type_iv_move(tour, dist, valid_indices, sample_size, device)
:canonical: src.models.policies.operators.unstringing.type_iv._find_best_type_iv_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv._find_best_type_iv_move
```
````

````{py:function} _evaluate_type_iv_move(tour, dist, i, j, l, k, N)
:canonical: src.models.policies.operators.unstringing.type_iv._evaluate_type_iv_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv._evaluate_type_iv_move
```
````

````{py:function} _apply_type_iv_move(tour: torch.Tensor, i: int, j: int, l: int, k: int) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_iv._apply_type_iv_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_iv._apply_type_iv_move
```
````
