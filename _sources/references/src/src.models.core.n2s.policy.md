# {py:mod}`src.models.core.n2s.policy`

```{py:module} src.models.core.n2s.policy
```

```{autodoc2-docstring} src.models.core.n2s.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`N2SPolicy <src.models.core.n2s.policy.N2SPolicy>`
  - ```{autodoc2-docstring} src.models.core.n2s.policy.N2SPolicy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`execute_n2s_request_move <src.models.core.n2s.policy.execute_n2s_request_move>`
  - ```{autodoc2-docstring} src.models.core.n2s.policy.execute_n2s_request_move
    :summary:
    ```
* - {py:obj}`validate_pdp_solution <src.models.core.n2s.policy.validate_pdp_solution>`
  - ```{autodoc2-docstring} src.models.core.n2s.policy.validate_pdp_solution
    :summary:
    ```
* - {py:obj}`create_feasible_pdp_solution <src.models.core.n2s.policy.create_feasible_pdp_solution>`
  - ```{autodoc2-docstring} src.models.core.n2s.policy.create_feasible_pdp_solution
    :summary:
    ```
* - {py:obj}`_update_n2s_history <src.models.core.n2s.policy._update_n2s_history>`
  - ```{autodoc2-docstring} src.models.core.n2s.policy._update_n2s_history
    :summary:
    ```
* - {py:obj}`_sync_n2s_caller <src.models.core.n2s.policy._sync_n2s_caller>`
  - ```{autodoc2-docstring} src.models.core.n2s.policy._sync_n2s_caller
    :summary:
    ```
````

### API

````{py:function} execute_n2s_request_move(solution: torch.Tensor, actions: torch.Tensor) -> torch.Tensor
:canonical: src.models.core.n2s.policy.execute_n2s_request_move

```{autodoc2-docstring} src.models.core.n2s.policy.execute_n2s_request_move
```
````

````{py:function} validate_pdp_solution(solution: torch.Tensor, partner_ids: torch.Tensor, is_pickup: torch.Tensor, is_delivery: torch.Tensor) -> None
:canonical: src.models.core.n2s.policy.validate_pdp_solution

```{autodoc2-docstring} src.models.core.n2s.policy.validate_pdp_solution
```
````

````{py:function} create_feasible_pdp_solution(bs: int, n: int, partner_ids: torch.Tensor, is_pickup: torch.Tensor, is_delivery: torch.Tensor, device: torch.device) -> torch.Tensor
:canonical: src.models.core.n2s.policy.create_feasible_pdp_solution

```{autodoc2-docstring} src.models.core.n2s.policy.create_feasible_pdp_solution
```
````

````{py:function} _update_n2s_history(history: torch.Tensor, move: torch.Tensor, window_queue: list[tuple[torch.Tensor, torch.Tensor]], window_size: int) -> None
:canonical: src.models.core.n2s.policy._update_n2s_history

```{autodoc2-docstring} src.models.core.n2s.policy._update_n2s_history
```
````

````{py:function} _sync_n2s_caller(orig_td: typing.Optional[tensordict.TensorDict], out: typing.Dict[str, typing.Any], bs: int, num_starts: int) -> None
:canonical: src.models.core.n2s.policy._sync_n2s_caller

```{autodoc2-docstring} src.models.core.n2s.policy._sync_n2s_caller
```
````

`````{py:class} N2SPolicy(embed_dim: int = 128, num_heads: int = 8, k_neighbors: int = 20, **kwargs: typing.Any)
:canonical: src.models.core.n2s.policy.N2SPolicy

Bases: {py:obj}`logic.src.models.common.improvement.policy.ImprovementPolicy`

```{autodoc2-docstring} src.models.core.n2s.policy.N2SPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.n2s.policy.N2SPolicy.__init__
```

````{py:method} _setup_initial_pdp_solution(td: tensordict.TensorDict, caller_had_solution: bool, caller_solution: typing.Any, has_explicit_pdp: bool, bs: int, n: int, device: torch.device, kwargs: dict[str, typing.Any]) -> None
:canonical: src.models.core.n2s.policy.N2SPolicy._setup_initial_pdp_solution

```{autodoc2-docstring} src.models.core.n2s.policy.N2SPolicy._setup_initial_pdp_solution
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Any = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: int | None = None, phase: str = 'train', return_actions: bool = True, **kwargs: typing.Any) -> dict[str, typing.Any]
:canonical: src.models.core.n2s.policy.N2SPolicy.forward

```{autodoc2-docstring} src.models.core.n2s.policy.N2SPolicy.forward
```

````

`````
