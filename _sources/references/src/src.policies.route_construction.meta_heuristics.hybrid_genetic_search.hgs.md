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

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_MIN_PENALTY <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MIN_PENALTY>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MIN_PENALTY
    :summary:
    ```
* - {py:obj}`_MAX_PENALTY <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MAX_PENALTY>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MAX_PENALTY
    :summary:
    ```
* - {py:obj}`_TARGET_COVERAGE_LOW <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_LOW>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_LOW
    :summary:
    ```
* - {py:obj}`_TARGET_COVERAGE_HIGH <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_HIGH>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_HIGH
    :summary:
    ```
* - {py:obj}`_TARGET_MARGIN <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_MARGIN>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_MARGIN
    :summary:
    ```
````

### API

````{py:data} _MIN_PENALTY
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MIN_PENALTY
:value: >
   0.05

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MIN_PENALTY
```

````

````{py:data} _MAX_PENALTY
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MAX_PENALTY
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._MAX_PENALTY
```

````

````{py:data} _TARGET_COVERAGE_LOW
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_LOW
:value: >
   0.4

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_LOW
```

````

````{py:data} _TARGET_COVERAGE_HIGH
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_HIGH
:value: >
   0.7

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_COVERAGE_HIGH
```

````

````{py:data} _TARGET_MARGIN
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_MARGIN
:value: >
   0.1

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs._TARGET_MARGIN
```

````

`````{py:class} HGSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params.HGSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, x_coords: typing.Optional[numpy.ndarray] = None, y_coords: typing.Optional[numpy.ndarray] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver.__init__
```

````{py:method} _evict_cache(ind: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, cache: dict, inv: typing.Dict[int, set]) -> None
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

````{py:method} _trim_pop(pop: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], cache: dict, inv: typing.Dict[int, set]) -> None
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

````{py:method} _record_offspring_signals(ind: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._record_offspring_signals

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver._record_offspring_signals
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
