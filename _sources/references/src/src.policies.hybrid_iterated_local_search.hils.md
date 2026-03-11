# {py:mod}`src.policies.hybrid_iterated_local_search.hils`

```{py:module} src.policies.hybrid_iterated_local_search.hils
```

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HILSSolver <src.policies.hybrid_iterated_local_search.hils.HILSSolver>`
  - ```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.hybrid_iterated_local_search.hils.logger>`
  - ```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.hybrid_iterated_local_search.hils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.logger
```

````

`````{py:class} HILSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.hybrid_iterated_local_search.params.HILSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver.__init__
```

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver.calculate_cost

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver.calculate_cost
```

````

````{py:method} calculate_profit(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver.calculate_profit

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver.calculate_profit
```

````

````{py:method} _add_to_pool(routes: typing.List[typing.List[int]])
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver._add_to_pool

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver._add_to_pool
```

````

````{py:method} _perturb(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver._perturb

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver._perturb
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver.build_initial_solution

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver.build_initial_solution
```

````

````{py:method} solve_set_partitioning() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver.solve_set_partitioning

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver.solve_set_partitioning
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_iterated_local_search.hils.HILSSolver.solve

```{autodoc2-docstring} src.policies.hybrid_iterated_local_search.hils.HILSSolver.solve
```

````

`````
