# {py:mod}`src.models.policies.operators.route.swap_star`

```{py:module} src.models.policies.operators.route.swap_star
```

```{autodoc2-docstring} src.models.policies.operators.route.swap_star
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_swap_star <src.models.policies.operators.route.swap_star.vectorized_swap_star>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.swap_star.vectorized_swap_star
    :summary:
    ```
* - {py:obj}`_identify_routes <src.models.policies.operators.route.swap_star._identify_routes>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.swap_star._identify_routes
    :summary:
    ```
* - {py:obj}`_compute_swap_star_gains <src.models.policies.operators.route.swap_star._compute_swap_star_gains>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.swap_star._compute_swap_star_gains
    :summary:
    ```
* - {py:obj}`_apply_swap_star_moves <src.models.policies.operators.route.swap_star._apply_swap_star_moves>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.swap_star._apply_swap_star_moves
    :summary:
    ```
````

### API

````{py:function} vectorized_swap_star(tours, dist_matrix, max_iterations=100)
:canonical: src.models.policies.operators.route.swap_star.vectorized_swap_star

```{autodoc2-docstring} src.models.policies.operators.route.swap_star.vectorized_swap_star
```
````

````{py:function} _identify_routes(tours, i, j, seq, B)
:canonical: src.models.policies.operators.route.swap_star._identify_routes

```{autodoc2-docstring} src.models.policies.operators.route.swap_star._identify_routes
```
````

````{py:function} _compute_swap_star_gains(tours, dist, node_i, node_j, i, j, start_i, end_i, start_j, end_j, b_idx, max_len, seq, device)
:canonical: src.models.policies.operators.route.swap_star._compute_swap_star_gains

```{autodoc2-docstring} src.models.policies.operators.route.swap_star._compute_swap_star_gains
```
````

````{py:function} _apply_swap_star_moves(tours, improved, i, j, ins_u, ins_v, max_len, device)
:canonical: src.models.policies.operators.route.swap_star._apply_swap_star_moves

```{autodoc2-docstring} src.models.policies.operators.route.swap_star._apply_swap_star_moves
```
````
