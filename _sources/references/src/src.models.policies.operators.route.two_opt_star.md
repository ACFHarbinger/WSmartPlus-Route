# {py:mod}`src.models.policies.operators.route.two_opt_star`

```{py:module} src.models.policies.operators.route.two_opt_star
```

```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_two_opt_star <src.models.policies.operators.route.two_opt_star.vectorized_two_opt_star>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star.vectorized_two_opt_star
    :summary:
    ```
* - {py:obj}`_identify_two_opt_star_routes <src.models.policies.operators.route.two_opt_star._identify_two_opt_star_routes>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._identify_two_opt_star_routes
    :summary:
    ```
* - {py:obj}`_compute_two_opt_star_gain <src.models.policies.operators.route.two_opt_star._compute_two_opt_star_gain>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._compute_two_opt_star_gain
    :summary:
    ```
* - {py:obj}`_apply_two_opt_star_moves <src.models.policies.operators.route.two_opt_star._apply_two_opt_star_moves>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._apply_two_opt_star_moves
    :summary:
    ```
* - {py:obj}`_map_tail_swap <src.models.policies.operators.route.two_opt_star._map_tail_swap>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._map_tail_swap
    :summary:
    ```
````

### API

````{py:function} vectorized_two_opt_star(tours, dist_matrix, max_iterations=200)
:canonical: src.models.policies.operators.route.two_opt_star.vectorized_two_opt_star

```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star.vectorized_two_opt_star
```
````

````{py:function} _identify_two_opt_star_routes(tours, i, j, seq, B)
:canonical: src.models.policies.operators.route.two_opt_star._identify_two_opt_star_routes

```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._identify_two_opt_star_routes
```
````

````{py:function} _compute_two_opt_star_gain(tours, dist, u, v, i, j, b_idx)
:canonical: src.models.policies.operators.route.two_opt_star._compute_two_opt_star_gain

```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._compute_two_opt_star_gain
```
````

````{py:function} _apply_two_opt_star_moves(tours, improved, i, j, end_i, end_j, max_len, seq, device)
:canonical: src.models.policies.operators.route.two_opt_star._apply_two_opt_star_moves

```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._apply_two_opt_star_moves
```
````

````{py:function} _map_tail_swap(idx_map, mask, i, j, end_i, end_j, B, max_len)
:canonical: src.models.policies.operators.route.two_opt_star._map_tail_swap

```{autodoc2-docstring} src.models.policies.operators.route.two_opt_star._map_tail_swap
```
````
