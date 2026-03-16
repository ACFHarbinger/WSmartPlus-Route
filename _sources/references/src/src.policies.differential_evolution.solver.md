# {py:mod}`src.policies.differential_evolution.solver`

```{py:module} src.policies.differential_evolution.solver
```

```{autodoc2-docstring} src.policies.differential_evolution.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DESolver <src.policies.differential_evolution.solver.DESolver>`
  - ```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver
    :summary:
    ```
````

### API

`````{py:class} DESolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.differential_evolution.params.DEParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.differential_evolution.solver.DESolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.differential_evolution.solver.DESolver.solve

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver.solve
```

````

````{py:method} _differential_mutation(base: typing.List[typing.List[int]], diff1: typing.List[typing.List[int]], diff2: typing.List[typing.List[int]], F: float) -> typing.List[typing.List[int]]
:canonical: src.policies.differential_evolution.solver.DESolver._differential_mutation

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._differential_mutation
```

````

````{py:method} _binomial_crossover(target: typing.List[typing.List[int]], mutant: typing.List[typing.List[int]], CR: float) -> typing.List[typing.List[int]]
:canonical: src.policies.differential_evolution.solver.DESolver._binomial_crossover

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._binomial_crossover
```

````

````{py:method} _initialize_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.differential_evolution.solver.DESolver._initialize_solution

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._initialize_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.differential_evolution.solver.DESolver._evaluate

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.differential_evolution.solver.DESolver._cost

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._cost
```

````

`````
