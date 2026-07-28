# {py:mod}`src.policies.vector.operators.destroy.shaw_removal`

```{py:module} src.policies.vector.operators.destroy.shaw_removal
```

```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_shaw_removal <src.policies.vector.operators.destroy.shaw_removal.vectorized_shaw_removal>`
  - ```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal.vectorized_shaw_removal
    :summary:
    ```
* - {py:obj}`_prepare_shaw_inputs <src.policies.vector.operators.destroy.shaw_removal._prepare_shaw_inputs>`
  - ```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._prepare_shaw_inputs
    :summary:
    ```
* - {py:obj}`_select_seed_nodes <src.policies.vector.operators.destroy.shaw_removal._select_seed_nodes>`
  - ```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._select_seed_nodes
    :summary:
    ```
* - {py:obj}`_calculate_relatedness_batch <src.policies.vector.operators.destroy.shaw_removal._calculate_relatedness_batch>`
  - ```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._calculate_relatedness_batch
    :summary:
    ```
* - {py:obj}`_select_next_removal_batch <src.policies.vector.operators.destroy.shaw_removal._select_next_removal_batch>`
  - ```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._select_next_removal_batch
    :summary:
    ```
* - {py:obj}`_init_shaw_params <src.policies.vector.operators.destroy.shaw_removal._init_shaw_params>`
  - ```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._init_shaw_params
    :summary:
    ```
````

### API

````{py:function} vectorized_shaw_removal(tours: torch.Tensor, distance_matrix: torch.Tensor, n_remove: int, wastes: typing.Optional[torch.Tensor] = None, time_windows: typing.Optional[torch.Tensor] = None, phi: float = 9.0, chi: float = 3.0, psi: float = 2.0, randomization_factor: float = 2.0, generator: typing.Optional[torch.Generator] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.policies.vector.operators.destroy.shaw_removal.vectorized_shaw_removal

```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal.vectorized_shaw_removal
```
````

````{py:function} _prepare_shaw_inputs(tours: torch.Tensor, distance_matrix: torch.Tensor, wastes: typing.Optional[torch.Tensor], time_windows: typing.Optional[torch.Tensor]) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor], typing.Optional[torch.Tensor], bool]
:canonical: src.policies.vector.operators.destroy.shaw_removal._prepare_shaw_inputs

```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._prepare_shaw_inputs
```
````

````{py:function} _select_seed_nodes(B: int, valid_mask: torch.Tensor, valid_counts: torch.Tensor, removed_mask: torch.Tensor, removed_list: torch.Tensor, removed_count: torch.Tensor, device: torch.device, generator: typing.Optional[torch.Generator]) -> None
:canonical: src.policies.vector.operators.destroy.shaw_removal._select_seed_nodes

```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._select_seed_nodes
```
````

````{py:function} _calculate_relatedness_batch(B: int, N: int, tours: torch.Tensor, distance_matrix: torch.Tensor, wastes: typing.Optional[torch.Tensor], time_windows: typing.Optional[torch.Tensor], removed_mask: torch.Tensor, removed_count: torch.Tensor, valid_counts: torch.Tensor, max_dist: float, max_waste: float, max_time: float, phi: float, psi: float, chi: float, device: torch.device) -> torch.Tensor
:canonical: src.policies.vector.operators.destroy.shaw_removal._calculate_relatedness_batch

```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._calculate_relatedness_batch
```
````

````{py:function} _select_next_removal_batch(B: int, n_remove: int, randomization_factor: float, relatedness_scores: torch.Tensor, removed_mask: torch.Tensor, removed_list: torch.Tensor, removed_count: torch.Tensor, valid_mask: torch.Tensor, valid_counts: torch.Tensor, device: torch.device, generator: typing.Optional[torch.Generator]) -> None
:canonical: src.policies.vector.operators.destroy.shaw_removal._select_next_removal_batch

```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._select_next_removal_batch
```
````

````{py:function} _init_shaw_params(tours: torch.Tensor, distance_matrix: torch.Tensor, wastes: typing.Optional[torch.Tensor], time_windows: typing.Optional[torch.Tensor]) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor], typing.Optional[torch.Tensor], float, float, float]
:canonical: src.policies.vector.operators.destroy.shaw_removal._init_shaw_params

```{autodoc2-docstring} src.policies.vector.operators.destroy.shaw_removal._init_shaw_params
```
````
