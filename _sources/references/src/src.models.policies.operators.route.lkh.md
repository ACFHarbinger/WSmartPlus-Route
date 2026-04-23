# {py:mod}`src.models.policies.operators.route.lkh`

```{py:module} src.models.policies.operators.route.lkh
```

```{autodoc2-docstring} src.models.policies.operators.route.lkh
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_lkh <src.models.policies.operators.route.lkh.vectorized_lkh>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh.vectorized_lkh
    :summary:
    ```
* - {py:obj}`_run_local_search <src.models.policies.operators.route.lkh._run_local_search>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._run_local_search
    :summary:
    ```
* - {py:obj}`_try_moves_for_node <src.models.policies.operators.route.lkh._try_moves_for_node>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._try_moves_for_node
    :summary:
    ```
* - {py:obj}`_compute_alpha_measures <src.models.policies.operators.route.lkh._compute_alpha_measures>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._compute_alpha_measures
    :summary:
    ```
* - {py:obj}`_get_candidate_sets <src.models.policies.operators.route.lkh._get_candidate_sets>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._get_candidate_sets
    :summary:
    ```
* - {py:obj}`_compute_score <src.models.policies.operators.route.lkh._compute_score>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._compute_score
    :summary:
    ```
* - {py:obj}`_is_better <src.models.policies.operators.route.lkh._is_better>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._is_better
    :summary:
    ```
* - {py:obj}`_apply_2opt <src.models.policies.operators.route.lkh._apply_2opt>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._apply_2opt
    :summary:
    ```
* - {py:obj}`_apply_3opt <src.models.policies.operators.route.lkh._apply_3opt>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._apply_3opt
    :summary:
    ```
* - {py:obj}`_double_bridge_kick <src.models.policies.operators.route.lkh._double_bridge_kick>`
  - ```{autodoc2-docstring} src.models.policies.operators.route.lkh._double_bridge_kick
    :summary:
    ```
````

### API

````{py:function} vectorized_lkh(tours: torch.Tensor, distance_matrix: torch.Tensor, capacities: typing.Optional[torch.Tensor] = None, wastes: typing.Optional[torch.Tensor] = None, max_iterations: int = 100, max_candidates: int = 5, use_3opt: bool = True, perturbation_interval: int = 10, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.route.lkh.vectorized_lkh

```{autodoc2-docstring} src.models.policies.operators.route.lkh.vectorized_lkh
```
````

````{py:function} _run_local_search(tour: torch.Tensor, dist: torch.Tensor, waste: typing.Optional[torch.Tensor], capacity: typing.Optional[torch.Tensor], candidates: typing.List[typing.List[int]], use_3opt: bool) -> torch.Tensor
:canonical: src.models.policies.operators.route.lkh._run_local_search

```{autodoc2-docstring} src.models.policies.operators.route.lkh._run_local_search
```
````

````{py:function} _try_moves_for_node(tour: torch.Tensor, i: int, nodes_count: int, dist: torch.Tensor, waste: typing.Optional[torch.Tensor], capacity: typing.Optional[torch.Tensor], candidates: typing.List[typing.List[int]], use_3opt: bool, curr_p: float, curr_c: float) -> typing.Tuple[bool, torch.Tensor, float, float]
:canonical: src.models.policies.operators.route.lkh._try_moves_for_node

```{autodoc2-docstring} src.models.policies.operators.route.lkh._try_moves_for_node
```
````

````{py:function} _compute_alpha_measures(distance_matrix: torch.Tensor, device: torch.device) -> torch.Tensor
:canonical: src.models.policies.operators.route.lkh._compute_alpha_measures

```{autodoc2-docstring} src.models.policies.operators.route.lkh._compute_alpha_measures
```
````

````{py:function} _get_candidate_sets(alpha_measures: torch.Tensor, max_candidates: int) -> typing.List[typing.List[int]]
:canonical: src.models.policies.operators.route.lkh._get_candidate_sets

```{autodoc2-docstring} src.models.policies.operators.route.lkh._get_candidate_sets
```
````

````{py:function} _compute_score(tour: torch.Tensor, distance_matrix: torch.Tensor, wastes: typing.Optional[torch.Tensor], capacity: typing.Optional[torch.Tensor]) -> typing.Tuple[float, float]
:canonical: src.models.policies.operators.route.lkh._compute_score

```{autodoc2-docstring} src.models.policies.operators.route.lkh._compute_score
```
````

````{py:function} _is_better(p1: float, c1: float, p2: float, c2: float) -> bool
:canonical: src.models.policies.operators.route.lkh._is_better

```{autodoc2-docstring} src.models.policies.operators.route.lkh._is_better
```
````

````{py:function} _apply_2opt(tour: torch.Tensor, i: int, j: int) -> torch.Tensor
:canonical: src.models.policies.operators.route.lkh._apply_2opt

```{autodoc2-docstring} src.models.policies.operators.route.lkh._apply_2opt
```
````

````{py:function} _apply_3opt(tour: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor
:canonical: src.models.policies.operators.route.lkh._apply_3opt

```{autodoc2-docstring} src.models.policies.operators.route.lkh._apply_3opt
```
````

````{py:function} _double_bridge_kick(tour: torch.Tensor, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.route.lkh._double_bridge_kick

```{autodoc2-docstring} src.models.policies.operators.route.lkh._double_bridge_kick
```
````
