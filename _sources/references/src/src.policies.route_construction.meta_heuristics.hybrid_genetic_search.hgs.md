# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSSolver <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver
    :summary:
    ```
````

### API

`````{py:class} HGSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params.HGSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, x_coords: typing.Optional[numpy.ndarray] = None, y_coords: typing.Optional[numpy.ndarray] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver.__init__
```

````{py:method} _evict_cache(ind: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, cache: dict) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._evict_cache

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._evict_cache
```

````

````{py:method} _insert_into_pop(ind: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, pop_feasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], pop_infeasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual]) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._insert_into_pop

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._insert_into_pop
```

````

````{py:method} _initialize_population(penalty_capacity: float) -> typing.Tuple[typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._initialize_population

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._initialize_population
```

````

````{py:method} _trim_pop(pop: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], cache: dict) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._trim_pop

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._trim_pop
```

````

````{py:method} _find_clone(pop: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual]) -> typing.Optional[int]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._find_clone

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._find_clone
```

````

````{py:method} _select_parents(pop_feasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], pop_infeasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual]) -> typing.Tuple[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._select_parents

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._select_parents
```

````

````{py:method} _generate_offspring(pop_feasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], pop_infeasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], penalty_capacity: float) -> src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._generate_offspring

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._generate_offspring
```

````

````{py:method} _insert_and_repair(child: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, pop_feasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], pop_infeasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], penalty_capacity: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._insert_and_repair

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._insert_and_repair
```

````

````{py:method} _adjust_penalties(current_penalty: float) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._adjust_penalties

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._adjust_penalties
```

````

````{py:method} _get_best_solution(pop_feasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], pop_infeasible: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual]) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._get_best_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._get_best_solution
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver.solve
```

````

`````
