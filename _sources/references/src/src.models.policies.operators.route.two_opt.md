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

````{py:function} vectorized_two_opt(tours: torch.Tensor, distance_matrix: torch.Tensor, max_iterations: int = 200, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.route.two_opt.vectorized_two_opt

```{autodoc2-docstring} src.models.policies.operators.route.two_opt.vectorized_two_opt
```
````

````{py:function} _compute_two_opt_gains(tours: torch.Tensor, dist: torch.Tensor, I_vals: torch.Tensor, J_vals: torch.Tensor, b_idx: torch.Tensor, B: int, K: int) -> torch.Tensor
:canonical: src.models.policies.operators.route.two_opt._compute_two_opt_gains

```{autodoc2-docstring} src.models.policies.operators.route.two_opt._compute_two_opt_gains
```
````

````{py:function} _apply_two_opt_moves(tours: torch.Tensor, improved: torch.Tensor, I_vals: torch.Tensor, J_vals: torch.Tensor, best_idx: torch.Tensor, B: int, N: int, device: torch.device) -> torch.Tensor
:canonical: src.models.policies.operators.route.two_opt._apply_two_opt_moves

```{autodoc2-docstring} src.models.policies.operators.route.two_opt._apply_two_opt_moves
```
````
