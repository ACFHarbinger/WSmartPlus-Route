# {py:mod}`src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco`

```{py:module} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperHeuristicACO <src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO
    :summary:
    ```
````

### API

`````{py:class} HyperHeuristicACO(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params.HyperACOParams] = None, initial_solution: typing.Optional[typing.List[typing.List[int]]] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.solve

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.solve
```

````

````{py:method} construct(nodes: typing.List[int], mandatory_nodes: typing.Optional[typing.List[int]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.construct

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.construct
```

````

````{py:method} _bootstrap(nodes: typing.List[int], effective_mandatory: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._bootstrap

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._bootstrap
```

````

````{py:method} _bootstrap_standard(nodes: typing.List[int], mandatory: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._bootstrap_standard

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._bootstrap_standard
```

````

````{py:method} _bootstrap_profit(nodes: typing.List[int], mandatory: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._bootstrap_profit

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._bootstrap_profit
```

````

````{py:method} build_solution(base_solution: typing.List[typing.List[int]], start_op_idx: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[str], numpy.ndarray, int]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.build_solution

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.build_solution
```

````

````{py:method} deposit_edge(i: int, j: int, delta: float) -> None
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.deposit_edge

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.deposit_edge
```

````

````{py:method} _make_context(routes: typing.List[typing.List[int]], effective_capacity: float, use_cache: bool = False) -> src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators.HyperOperatorContext
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._make_context

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._make_context
```

````

````{py:method} _select_sequence(start_op_idx: int) -> typing.Tuple[typing.List[str], int]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._select_sequence

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._select_sequence
```

````

````{py:method} _calculate_routing_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._calculate_routing_cost

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._calculate_routing_cost
```

````

````{py:method} _evaluate_objective(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._evaluate_objective

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._evaluate_objective
```

````

````{py:method} _evaporate_pheromones() -> None
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._evaporate_pheromones

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._evaporate_pheromones
```

````

````{py:method} evaporate_all() -> None
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.evaporate_all

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.evaporate_all
```

````

`````
