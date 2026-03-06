# {py:mod}`src.models.policies.operators.move.swap`

```{py:module} src.models.policies.operators.move.swap
```

```{autodoc2-docstring} src.models.policies.operators.move.swap
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_swap <src.models.policies.operators.move.swap.vectorized_swap>`
  - ```{autodoc2-docstring} src.models.policies.operators.move.swap.vectorized_swap
    :summary:
    ```
* - {py:obj}`_compute_swap_gain <src.models.policies.operators.move.swap._compute_swap_gain>`
  - ```{autodoc2-docstring} src.models.policies.operators.move.swap._compute_swap_gain
    :summary:
    ```
* - {py:obj}`_apply_swap_moves <src.models.policies.operators.move.swap._apply_swap_moves>`
  - ```{autodoc2-docstring} src.models.policies.operators.move.swap._apply_swap_moves
    :summary:
    ```
````

### API

````{py:function} vectorized_swap(tours, dist_matrix, max_iterations=200, generator=None)
:canonical: src.models.policies.operators.move.swap.vectorized_swap

```{autodoc2-docstring} src.models.policies.operators.move.swap.vectorized_swap
```
````

````{py:function} _compute_swap_gain(tours, dist, node_i, node_j, i, j, b_idx)
:canonical: src.models.policies.operators.move.swap._compute_swap_gain

```{autodoc2-docstring} src.models.policies.operators.move.swap._compute_swap_gain
```
````

````{py:function} _apply_swap_moves(tours, improved, i, j)
:canonical: src.models.policies.operators.move.swap._apply_swap_moves

```{autodoc2-docstring} src.models.policies.operators.move.swap._apply_swap_moves
```
````
