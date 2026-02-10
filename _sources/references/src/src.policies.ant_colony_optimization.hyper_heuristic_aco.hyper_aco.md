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

`````{py:class} HyperHeuristicACO(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, heuristic_matrix: numpy.ndarray, problem: str = 'vrpp', params: typing.Optional[src.policies.ant_colony_optimization.hyper_heuristic_aco.params.HyperACOParams] = None, **kwargs)
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.__init__
```

````{py:method} _init_pheromones()
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._init_pheromones

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._init_pheromones
```

````

````{py:method} solve(initial_routes: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.solve

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO.solve
```

````

````{py:method} _construct_sequence() -> typing.List[str]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._construct_sequence

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._construct_sequence
```

````

````{py:method} _select_next_operator(current: int) -> int
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._select_next_operator

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._select_next_operator
```

````

````{py:method} _apply_sequence(routes: typing.List[typing.List[int]], sequence: typing.List[str]) -> typing.List[typing.List[int]]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._apply_sequence

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._apply_sequence
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

````{py:method} _deposit_pheromones(solutions: typing.List[typing.Tuple[typing.List[typing.List[int]], float, typing.List[str]]])
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._deposit_pheromones

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._deposit_pheromones
```

````

````{py:method} _update_heuristics()
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._update_heuristics

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_aco.HyperHeuristicACO._update_heuristics
```

````

`````
