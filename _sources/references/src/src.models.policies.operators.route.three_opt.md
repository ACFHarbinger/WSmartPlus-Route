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

````{py:function} vectorized_three_opt(tours: torch.Tensor, dist_matrix: torch.Tensor, max_iterations: int = 100, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.route.three_opt.vectorized_three_opt

```{autodoc2-docstring} src.models.policies.operators.route.three_opt.vectorized_three_opt
```
````

````{py:function} _compute_three_opt_gains(tours: torch.Tensor, dist: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor, b_idx: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.operators.route.three_opt._compute_three_opt_gains

```{autodoc2-docstring} src.models.policies.operators.route.three_opt._compute_three_opt_gains
```
````

````{py:function} _apply_three_opt_moves(tours: torch.Tensor, improved: torch.Tensor, best_case: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor
:canonical: src.models.policies.operators.route.three_opt._apply_three_opt_moves

```{autodoc2-docstring} src.models.policies.operators.route.three_opt._apply_three_opt_moves
```
````
