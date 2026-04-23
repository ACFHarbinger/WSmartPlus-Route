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

````{py:function} vectorized_swap_star(tours: torch.Tensor, dist_matrix: torch.Tensor, max_iterations: int = 100, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.route.swap_star.vectorized_swap_star

```{autodoc2-docstring} src.models.policies.operators.route.swap_star.vectorized_swap_star
```
````

````{py:function} _identify_routes(tours: torch.Tensor, i: torch.Tensor, j: torch.Tensor, seq: torch.Tensor, B: int) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.policies.operators.route.swap_star._identify_routes

```{autodoc2-docstring} src.models.policies.operators.route.swap_star._identify_routes
```
````

````{py:function} _compute_swap_star_gains(tours: torch.Tensor, dist: torch.Tensor, node_i: torch.Tensor, node_j: torch.Tensor, i: torch.Tensor, j: torch.Tensor, start_i: torch.Tensor, end_i: torch.Tensor, start_j: torch.Tensor, end_j: torch.Tensor, b_idx: torch.Tensor, max_len: int, seq: torch.Tensor, device: torch.device) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.policies.operators.route.swap_star._compute_swap_star_gains

```{autodoc2-docstring} src.models.policies.operators.route.swap_star._compute_swap_star_gains
```
````

````{py:function} _apply_swap_star_moves(tours: torch.Tensor, improved: torch.Tensor, i: torch.Tensor, j: torch.Tensor, ins_u: torch.Tensor, ins_v: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor
:canonical: src.models.policies.operators.route.swap_star._apply_swap_star_moves

```{autodoc2-docstring} src.models.policies.operators.route.swap_star._apply_swap_star_moves
```
````
