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

````{py:function} vectorized_relocate(tours: torch.Tensor, dist_matrix: torch.Tensor, max_iterations: int = 200, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.move.relocate.vectorized_relocate

```{autodoc2-docstring} src.models.policies.operators.move.relocate.vectorized_relocate
```
````

````{py:function} _compute_relocate_gain(tours: torch.Tensor, dist: torch.Tensor, node_i: torch.Tensor, i: torch.Tensor, j: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor
:canonical: src.models.policies.operators.move.relocate._compute_relocate_gain

```{autodoc2-docstring} src.models.policies.operators.move.relocate._compute_relocate_gain
```
````

````{py:function} _apply_relocate_move(tours: torch.Tensor, improved: torch.Tensor, i: torch.Tensor, j: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor
:canonical: src.models.policies.operators.move.relocate._apply_relocate_move

```{autodoc2-docstring} src.models.policies.operators.move.relocate._apply_relocate_move
```
````
