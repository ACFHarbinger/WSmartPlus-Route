# {py:mod}`src.models.policies.operators.exchange.or_opt`

```{py:module} src.models.policies.operators.exchange.or_opt
```

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_or_opt <src.models.policies.operators.exchange.or_opt.vectorized_or_opt>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt.vectorized_or_opt
    :summary:
    ```
* - {py:obj}`_compute_removal_info <src.models.policies.operators.exchange.or_opt._compute_removal_info>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._compute_removal_info
    :summary:
    ```
* - {py:obj}`_find_best_insertions <src.models.policies.operators.exchange.or_opt._find_best_insertions>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._find_best_insertions
    :summary:
    ```
* - {py:obj}`_apply_or_opt_moves <src.models.policies.operators.exchange.or_opt._apply_or_opt_moves>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._apply_or_opt_moves
    :summary:
    ```
````

### API

````{py:function} vectorized_or_opt(tours: torch.Tensor, distance_matrix: torch.Tensor, capacities: typing.Optional[torch.Tensor] = None, wastes: typing.Optional[torch.Tensor] = None, chain_lengths: typing.Tuple[int, ...] = (1, 2, 3), max_iterations: int = 100, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.exchange.or_opt.vectorized_or_opt

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt.vectorized_or_opt
```
````

````{py:function} _compute_removal_info(tours: torch.Tensor, chain_starts: torch.Tensor, chain_len: int, dist_mat: torch.Tensor, wastes: typing.Optional[torch.Tensor], has_capacity: bool, batch_indices: torch.Tensor, B: int, n_chains: int, N: int) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]
:canonical: src.models.policies.operators.exchange.or_opt._compute_removal_info

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._compute_removal_info
```
````

````{py:function} _find_best_insertions(B: int, n_chains: int, N: int, chain_len: int, chain_starts: torch.Tensor, tours: torch.Tensor, dist_mat: torch.Tensor, rem_gain: torch.Tensor, chain_wastes: typing.Optional[torch.Tensor], caps: typing.Optional[torch.Tensor], has_cap: bool, b_idx: torch.Tensor, device: torch.device) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.operators.exchange.or_opt._find_best_insertions

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._find_best_insertions
```
````

````{py:function} _apply_or_opt_moves(tours: torch.Tensor, improved: torch.Tensor, best_chain_idx: torch.Tensor, best_insert_pos: torch.Tensor, chain_starts: torch.Tensor, chain_len: int, B: int, N: int, device: torch.device) -> torch.Tensor
:canonical: src.models.policies.operators.exchange.or_opt._apply_or_opt_moves

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._apply_or_opt_moves
```
````
