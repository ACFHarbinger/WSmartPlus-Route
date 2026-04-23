# {py:mod}`src.models.policies.operators.exchange.ejection_chain`

```{py:module} src.models.policies.operators.exchange.ejection_chain
```

```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_ejection_chain <src.models.policies.operators.exchange.ejection_chain.vectorized_ejection_chain>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain.vectorized_ejection_chain
    :summary:
    ```
* - {py:obj}`_try_insert_with_ejection_chain <src.models.policies.operators.exchange.ejection_chain._try_insert_with_ejection_chain>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._try_insert_with_ejection_chain
    :summary:
    ```
* - {py:obj}`_get_routes_with_loads <src.models.policies.operators.exchange.ejection_chain._get_routes_with_loads>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._get_routes_with_loads
    :summary:
    ```
* - {py:obj}`_attempt_route_elimination <src.models.policies.operators.exchange.ejection_chain._attempt_route_elimination>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._attempt_route_elimination
    :summary:
    ```
* - {py:obj}`_find_best_insertion_in_route <src.models.policies.operators.exchange.ejection_chain._find_best_insertion_in_route>`
  - ```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._find_best_insertion_in_route
    :summary:
    ```
````

### API

````{py:function} vectorized_ejection_chain(tours: torch.Tensor, distance_matrix: torch.Tensor, capacities: typing.Optional[torch.Tensor] = None, wastes: typing.Optional[torch.Tensor] = None, max_depth: int = 5, target_route_reduction: typing.Optional[int] = None, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.models.policies.operators.exchange.ejection_chain.vectorized_ejection_chain

```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain.vectorized_ejection_chain
```
````

````{py:function} _try_insert_with_ejection_chain(tour: torch.Tensor, node: int, source_start: int, source_end: int, dist: torch.Tensor, waste: torch.Tensor, capacity: torch.Tensor, max_depth: int, ejection_log: typing.List[typing.Tuple[int, int]], depot_positions: typing.List[int]) -> bool
:canonical: src.models.policies.operators.exchange.ejection_chain._try_insert_with_ejection_chain

```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._try_insert_with_ejection_chain
```
````

````{py:function} _get_routes_with_loads(tour: torch.Tensor, waste: torch.Tensor, N: int) -> typing.List[typing.Tuple[int, int, typing.List[int], float]]
:canonical: src.models.policies.operators.exchange.ejection_chain._get_routes_with_loads

```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._get_routes_with_loads
```
````

````{py:function} _attempt_route_elimination(tour: torch.Tensor, route_data: typing.Tuple[int, int, typing.List[int], float], dist: torch.Tensor, waste: torch.Tensor, capacity: torch.Tensor, max_depth: int, depot_positions: typing.List[int]) -> typing.Tuple[bool, torch.Tensor]
:canonical: src.models.policies.operators.exchange.ejection_chain._attempt_route_elimination

```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._attempt_route_elimination
```
````

````{py:function} _find_best_insertion_in_route(tour: torch.Tensor, node: int, start: int, end: int, dist: torch.Tensor) -> int
:canonical: src.models.policies.operators.exchange.ejection_chain._find_best_insertion_in_route

```{autodoc2-docstring} src.models.policies.operators.exchange.ejection_chain._find_best_insertion_in_route
```
````
