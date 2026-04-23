# {py:mod}`src.models.policies.operators.unstringing.type_i`

```{py:module} src.models.policies.operators.unstringing.type_i
```

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_type_i_unstringing <src.models.policies.operators.unstringing.type_i.vectorized_type_i_unstringing>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i.vectorized_type_i_unstringing
    :summary:
    ```
* - {py:obj}`_find_best_type_i_move <src.models.policies.operators.unstringing.type_i._find_best_type_i_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i._find_best_type_i_move
    :summary:
    ```
* - {py:obj}`_evaluate_type_i_move <src.models.policies.operators.unstringing.type_i._evaluate_type_i_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i._evaluate_type_i_move
    :summary:
    ```
* - {py:obj}`_apply_type_i_move <src.models.policies.operators.unstringing.type_i._apply_type_i_move>`
  - ```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i._apply_type_i_move
    :summary:
    ```
````

### API

````{py:function} vectorized_type_i_unstringing(tours: torch.Tensor, distance_matrix: torch.Tensor, max_iterations: int = 50, sample_size: int = 100, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_i.vectorized_type_i_unstringing

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i.vectorized_type_i_unstringing
```
````

````{py:function} _find_best_type_i_move(tour: torch.Tensor, dist: torch.Tensor, valid_indices: torch.Tensor, sample_size: int, device: torch.device, generator: typing.Optional[torch.Generator] = None) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, int]]]
:canonical: src.models.policies.operators.unstringing.type_i._find_best_type_i_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i._find_best_type_i_move
```
````

````{py:function} _evaluate_type_i_move(tour: torch.Tensor, dist: torch.Tensor, i: int, j: int, k: int, N: int) -> float
:canonical: src.models.policies.operators.unstringing.type_i._evaluate_type_i_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i._evaluate_type_i_move
```
````

````{py:function} _apply_type_i_move(tour: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor
:canonical: src.models.policies.operators.unstringing.type_i._apply_type_i_move

```{autodoc2-docstring} src.models.policies.operators.unstringing.type_i._apply_type_i_move
```
````
