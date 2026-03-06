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

````{py:function} vectorized_or_opt(tours: torch.Tensor, distance_matrix: torch.Tensor, capacities: typing.Optional[torch.Tensor] = None, wastes: typing.Optional[torch.Tensor] = None, chain_lengths: tuple = (1, 2, 3), max_iterations: int = 100, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.exchange.or_opt.vectorized_or_opt

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt.vectorized_or_opt
```
````

````{py:function} _compute_removal_info(tours, chain_starts, chain_len, dist_mat, wastes, has_capacity, batch_indices, B, n_chains, N)
:canonical: src.models.policies.operators.exchange.or_opt._compute_removal_info

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._compute_removal_info
```
````

````{py:function} _find_best_insertions(B, n_chains, N, chain_len, chain_starts, tours, dist_mat, rem_gain, chain_wastes, caps, has_cap, b_idx, device)
:canonical: src.models.policies.operators.exchange.or_opt._find_best_insertions

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._find_best_insertions
```
````

````{py:function} _apply_or_opt_moves(tours, improved, best_chain_idx, best_insert_pos, chain_starts, chain_len, B, N, device)
:canonical: src.models.policies.operators.exchange.or_opt._apply_or_opt_moves

```{autodoc2-docstring} src.models.policies.operators.exchange.or_opt._apply_or_opt_moves
```
````
