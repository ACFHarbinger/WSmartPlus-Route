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

````{py:method} build_solution(base_solution: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[str], numpy.ndarray]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.build_solution

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO.build_solution
```

````

````{py:method} _select_sequence() -> typing.List[str]
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._select_sequence

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._select_sequence
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._calculate_cost

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._calculate_cost
```

````

````{py:method} _evaporate_pheromones()
:canonical: src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._evaporate_pheromones

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco.HyperHeuristicACO._evaporate_pheromones
```

````

`````
