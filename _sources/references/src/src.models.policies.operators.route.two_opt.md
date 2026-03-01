# {py:mod}`src.models.policies.operators.route.two_opt`

```{py:module} src.models.policies.operators.route.two_opt
```

```{autodoc2-docstring} src.models.policies.operators.route.two_opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_two_opt <src.models.policies.operators.route.two_opt.vectorized_two_opt>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt.vectorized_two_opt
    :summary:
    ```
* - {py:obj}`_compute_two_opt_gains <src.models.policies.operators.route.two_opt._compute_two_opt_gains>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt._compute_two_opt_gains
    :summary:
    ```
* - {py:obj}`_apply_two_opt_moves <src.models.policies.operators.route.two_opt._apply_two_opt_moves>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt._apply_two_opt_moves
    :summary:
    ```
````

### API

````{py:function} vectorized_two_opt(tours, distance_matrix, max_iterations=200)
:canonical: src.models.policies.operators.route.two_opt.vectorized_two_opt

```{autodoc2-docstring} src.models.policies.operators.route.two_opt.vectorized_two_opt
```
````

````{py:function} _compute_two_opt_gains(tours, dist, I_vals, J_vals, b_idx, B, K)
:canonical: src.models.policies.operators.route.two_opt._compute_two_opt_gains

```{autodoc2-docstring} src.models.policies.operators.route.two_opt._compute_two_opt_gains
```
````

````{py:function} _apply_two_opt_moves(tours, improved, I_vals, J_vals, best_idx, B, N, device)
:canonical: src.models.policies.operators.route.two_opt._apply_two_opt_moves

```{autodoc2-docstring} src.models.policies.operators.route.two_opt._apply_two_opt_moves
```
````
