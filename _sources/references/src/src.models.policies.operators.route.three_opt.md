# {py:mod}`src.models.policies.operators.route.three_opt`

```{py:module} src.models.policies.operators.route.three_opt
```

```{autodoc2-docstring} src.models.policies.operators.route.three_opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_three_opt <src.models.policies.operators.route.three_opt.vectorized_three_opt>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.three_opt.vectorized_three_opt
    :summary:
    ```
* - {py:obj}`_compute_three_opt_gains <src.models.policies.operators.route.three_opt._compute_three_opt_gains>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.three_opt._compute_three_opt_gains
    :summary:
    ```
* - {py:obj}`_apply_three_opt_moves <src.models.policies.operators.route.three_opt._apply_three_opt_moves>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.three_opt._apply_three_opt_moves
    :summary:
    ```
````

### API

````{py:function} vectorized_three_opt(tours, dist_matrix, max_iterations=100, generator=None)
:canonical: src.models.policies.operators.route.three_opt.vectorized_three_opt

```{autodoc2-docstring} src.models.policies.operators.route.three_opt.vectorized_three_opt
```
````

````{py:function} _compute_three_opt_gains(tours, dist, i, j, k, b_idx)
:canonical: src.models.policies.operators.route.three_opt._compute_three_opt_gains

```{autodoc2-docstring} src.models.policies.operators.route.three_opt._compute_three_opt_gains
```
````

````{py:function} _apply_three_opt_moves(tours, improved, best_case, i, j, k, max_len, device)
:canonical: src.models.policies.operators.route.three_opt._apply_three_opt_moves

```{autodoc2-docstring} src.models.policies.operators.route.three_opt._apply_three_opt_moves
```
````
