# {py:mod}`src.policies.hybrid_volleyball_premier_league.hvpl`

```{py:module} src.policies.hybrid_volleyball_premier_league.hvpl
```

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HVPLSolver <src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver>`
  - ```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver
    :summary:
    ```
````

### API

`````{py:class} HVPLSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hybrid_volleyball_premier_league.params.HVPLParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver.solve

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver.solve
```

````

````{py:method} _canonicalize_routes(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._canonicalize_routes

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._canonicalize_routes
```

````

````{py:method} _hash_routes(routes: typing.List[typing.List[int]]) -> str
:canonical: src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._hash_routes

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._hash_routes
```

````

````{py:method} _get_best(population: typing.List[typing.Tuple[typing.List[typing.List[int]], float, float]]) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._get_best

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._get_best
```

````

````{py:method} _update_pheromones(routes: typing.List[typing.List[int]], cost: float) -> None
:canonical: src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._update_pheromones

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._update_pheromones
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._calculate_cost

```{autodoc2-docstring} src.policies.hybrid_volleyball_premier_league.hvpl.HVPLSolver._calculate_cost
```

````

`````
