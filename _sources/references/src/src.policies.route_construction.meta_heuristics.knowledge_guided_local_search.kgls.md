# {py:mod}`src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls`

```{py:module} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KGLSSolver <src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.logger
```

````

`````{py:class} KGLSSolver(dist_matrix: numpy.ndarray, locations: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.params.KGLSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.__init__
```

````{py:method} calculate_cost(routes: typing.List[typing.List[int]], penalized: bool = False) -> float
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.calculate_cost
```

````

````{py:method} calculate_profit(routes: typing.List[typing.List[int]], penalized: bool = False) -> float
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.calculate_profit

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.calculate_profit
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.build_initial_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.build_initial_solution
```

````

````{py:method} apply_local_search(routes: typing.List[typing.List[int]], ls_manager: logic.src.policies.helpers.local_search.local_search_aco.ACOLocalSearch, targeted_nodes: typing.Optional[typing.List[int]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.apply_local_search

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.apply_local_search
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.knowledge_guided_local_search.kgls.KGLSSolver.solve
```

````

`````
