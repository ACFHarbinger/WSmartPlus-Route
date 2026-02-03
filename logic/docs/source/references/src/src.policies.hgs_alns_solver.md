# {py:mod}`src.policies.hgs_alns_solver`

```{py:module} src.policies.hgs_alns_solver
```

```{autodoc2-docstring} src.policies.hgs_alns_solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSALNSSolver <src.policies.hgs_alns_solver.HGSALNSSolver>`
  - ```{autodoc2-docstring} src.policies.hgs_alns_solver.HGSALNSSolver
    :summary:
    ```
````

### API

`````{py:class} HGSALNSSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hgs_aux.types.HGSParams, alns_education_iterations: int = 50)
:canonical: src.policies.hgs_alns_solver.HGSALNSSolver

Bases: {py:obj}`src.policies.hybrid_genetic_search.HGSSolver`

```{autodoc2-docstring} src.policies.hgs_alns_solver.HGSALNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hgs_alns_solver.HGSALNSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hgs_alns_solver.HGSALNSSolver.solve

```{autodoc2-docstring} src.policies.hgs_alns_solver.HGSALNSSolver.solve
```

````

`````
