# {py:mod}`src.models.policies.operators.move.relocate`

```{py:module} src.models.policies.operators.move.relocate
```

```{autodoc2-docstring} src.models.policies.operators.move.relocate
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_relocate <src.models.policies.operators.move.relocate.vectorized_relocate>`
  - ```{autodoc2-docstring} src.models.policies.operators.move.relocate.vectorized_relocate
    :summary:
    ```
* - {py:obj}`_compute_relocate_gain <src.models.policies.operators.move.relocate._compute_relocate_gain>`
  - ```{autodoc2-docstring} src.models.policies.operators.move.relocate._compute_relocate_gain
    :summary:
    ```
* - {py:obj}`_apply_relocate_move <src.models.policies.operators.move.relocate._apply_relocate_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.move.relocate._apply_relocate_move
    :summary:
    ```
````

### API

````{py:function} vectorized_relocate(tours, dist_matrix, max_iterations=200)
:canonical: src.models.policies.operators.move.relocate.vectorized_relocate

```{autodoc2-docstring} src.models.policies.operators.move.relocate.vectorized_relocate
```
````

````{py:function} _compute_relocate_gain(tours, dist, node_i, i, j, b_idx)
:canonical: src.models.policies.operators.move.relocate._compute_relocate_gain

```{autodoc2-docstring} src.models.policies.operators.move.relocate._compute_relocate_gain
```
````

````{py:function} _apply_relocate_move(tours, improved, i, j, max_len, device)
:canonical: src.models.policies.operators.move.relocate._apply_relocate_move

```{autodoc2-docstring} src.models.policies.operators.move.relocate._apply_relocate_move
```
````
