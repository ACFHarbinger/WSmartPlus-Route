# {py:mod}`src.models.policies.operators.exchange.cross_exchange`

```{py:module} src.models.policies.operators.exchange.cross_exchange
```

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_cross_exchange <src.models.policies.operators.exchange.cross_exchange.vectorized_cross_exchange>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange.vectorized_cross_exchange
    :summary:
    ```
* - {py:obj}`_perform_cross_exchange_iteration <src.models.policies.operators.exchange.cross_exchange._perform_cross_exchange_iteration>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._perform_cross_exchange_iteration
    :summary:
    ```
* - {py:obj}`_get_routes_from_tour <src.models.policies.operators.exchange.cross_exchange._get_routes_from_tour>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._get_routes_from_tour
    :summary:
    ```
* - {py:obj}`_find_best_move_for_segments <src.models.policies.operators.exchange.cross_exchange._find_best_move_for_segments>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._find_best_move_for_segments
    :summary:
    ```
* - {py:obj}`_check_cross_capacity <src.models.policies.operators.exchange.cross_exchange._check_cross_capacity>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._check_cross_capacity
    :summary:
    ```
* - {py:obj}`_compute_cross_delta <src.models.policies.operators.exchange.cross_exchange._compute_cross_delta>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._compute_cross_delta
    :summary:
    ```
* - {py:obj}`_apply_cross_exchange_move <src.models.policies.operators.exchange.cross_exchange._apply_cross_exchange_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._apply_cross_exchange_move
    :summary:
    ```
````

### API

````{py:function} vectorized_cross_exchange(tours: torch.Tensor, distance_matrix: torch.Tensor, capacities: typing.Optional[torch.Tensor] = None, wastes: typing.Optional[torch.Tensor] = None, max_segment_len: int = 3, max_iterations: int = 50, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.exchange.cross_exchange.vectorized_cross_exchange

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange.vectorized_cross_exchange
```
````

````{py:function} _perform_cross_exchange_iteration(B, tours, max_segment_len, distance_matrix, wastes, capacities, has_capacity, device) -> bool
:canonical: src.models.policies.operators.exchange.cross_exchange._perform_cross_exchange_iteration

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._perform_cross_exchange_iteration
```
````

````{py:function} _get_routes_from_tour(tour: torch.Tensor)
:canonical: src.models.policies.operators.exchange.cross_exchange._get_routes_from_tour

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._get_routes_from_tour
```
````

````{py:function} _find_best_move_for_segments(b_idx, tour, routes, seg_a_len, seg_b_len, distance_matrix, wastes, capacities, has_capacity, device)
:canonical: src.models.policies.operators.exchange.cross_exchange._find_best_move_for_segments

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._find_best_move_for_segments
```
````

````{py:function} _check_cross_capacity(b, tour, s_a, len_a, s_b, len_b, r_a_s, r_a_e, r_b_s, r_b_e, wastes, capacities)
:canonical: src.models.policies.operators.exchange.cross_exchange._check_cross_capacity

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._check_cross_capacity
```
````

````{py:function} _compute_cross_delta(b, tour, s_a, len_a, s_b, len_b, r_a_s, r_a_e, r_b_s, r_b_e, dist_mat)
:canonical: src.models.policies.operators.exchange.cross_exchange._compute_cross_delta

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._compute_cross_delta
```
````

````{py:function} _apply_cross_exchange_move(tour, move, device)
:canonical: src.models.policies.operators.exchange.cross_exchange._apply_cross_exchange_move

```{autodoc2-docstring} src.models.policies.operators.exchange.cross_exchange._apply_cross_exchange_move
```
````
