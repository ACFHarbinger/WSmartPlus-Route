# {py:mod}`src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco`

```{py:module} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperHeuristicACO <src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO
    :summary:
    ```
````

### API

`````{py:class} HyperHeuristicACO(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams] = None, initial_solution: typing.Optional[typing.List[typing.List[int]]] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.solve

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.solve
```

````

````{py:method} build_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.build_solution

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.build_solution
```

````

````{py:method} _select_sequence() -> typing.List[str]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._select_sequence

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._select_sequence
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._calculate_cost

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._calculate_cost
```

````

````{py:method} _evaporate_pheromones()
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._evaporate_pheromones

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._evaporate_pheromones
```

````

````{py:method} _update_heuristics()
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._update_heuristics

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._update_heuristics
```

````

`````
