# {py:mod}`src.policies.other.operators.crossover.selective_route_exchange`

```{py:module} src.policies.other.operators.crossover.selective_route_exchange
```

```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_select_initial_child_routes <src.policies.other.operators.crossover.selective_route_exchange._select_initial_child_routes>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._select_initial_child_routes
    :summary:
    ```
* - {py:obj}`_merge_non_conflicting_p2_routes <src.policies.other.operators.crossover.selective_route_exchange._merge_non_conflicting_p2_routes>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._merge_non_conflicting_p2_routes
    :summary:
    ```
* - {py:obj}`_insert_missing_nodes_greedy <src.policies.other.operators.crossover.selective_route_exchange._insert_missing_nodes_greedy>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._insert_missing_nodes_greedy
    :summary:
    ```
* - {py:obj}`_enforce_mandatory_nodes_srex <src.policies.other.operators.crossover.selective_route_exchange._enforce_mandatory_nodes_srex>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._enforce_mandatory_nodes_srex
    :summary:
    ```
* - {py:obj}`selective_route_exchange_crossover <src.policies.other.operators.crossover.selective_route_exchange.selective_route_exchange_crossover>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange.selective_route_exchange_crossover
    :summary:
    ```
````

### API

````{py:function} _select_initial_child_routes(p1: logic.src.policies.hybrid_genetic_search.individual.Individual, rng: random.Random) -> typing.Tuple[typing.List[typing.List[int]], typing.Set[int]]
:canonical: src.policies.other.operators.crossover.selective_route_exchange._select_initial_child_routes

```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._select_initial_child_routes
```
````

````{py:function} _merge_non_conflicting_p2_routes(p2: logic.src.policies.hybrid_genetic_search.individual.Individual, child_routes: typing.List[typing.List[int]], child_nodes: typing.Set[int]) -> typing.List[int]
:canonical: src.policies.other.operators.crossover.selective_route_exchange._merge_non_conflicting_p2_routes

```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._merge_non_conflicting_p2_routes
```
````

````{py:function} _insert_missing_nodes_greedy(child_routes: typing.List[typing.List[int]], child_nodes: typing.Set[int], missing: typing.List[int], dist_matrix: typing.Optional[numpy.ndarray], wastes: typing.Optional[typing.Dict[int, float]], capacity: float, R: float, C: float, rng: random.Random) -> None
:canonical: src.policies.other.operators.crossover.selective_route_exchange._insert_missing_nodes_greedy

```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._insert_missing_nodes_greedy
```
````

````{py:function} _enforce_mandatory_nodes_srex(child_routes: typing.List[typing.List[int]], mandatory_nodes: typing.List[int], wastes: typing.Optional[typing.Dict[int, float]], capacity: float) -> None
:canonical: src.policies.other.operators.crossover.selective_route_exchange._enforce_mandatory_nodes_srex

```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange._enforce_mandatory_nodes_srex
```
````

````{py:function} selective_route_exchange_crossover(p1: logic.src.policies.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.hybrid_genetic_search.individual.Individual, rng: typing.Optional[random.Random] = None, dist_matrix: typing.Optional[numpy.ndarray] = None, wastes: typing.Optional[typing.Dict[int, float]] = None, capacity: float = float('inf'), R: float = 1.0, C: float = 1.0, mandatory_nodes: typing.Optional[typing.List[int]] = None) -> logic.src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.other.operators.crossover.selective_route_exchange.selective_route_exchange_crossover

```{autodoc2-docstring} src.policies.other.operators.crossover.selective_route_exchange.selective_route_exchange_crossover
```
````
