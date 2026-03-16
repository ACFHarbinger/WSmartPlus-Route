# {py:mod}`src.policies.improving_and_equal.solver`

```{py:module} src.policies.improving_and_equal.solver
```

```{autodoc2-docstring} src.policies.improving_and_equal.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IESolver <src.policies.improving_and_equal.solver.IESolver>`
  - ```{autodoc2-docstring} src.policies.improving_and_equal.solver.IESolver
    :summary:
    ```
````

### API

`````{py:class} IESolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.improving_and_equal.solver.IESolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.improving_and_equal.solver.IESolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.improving_and_equal.solver.IESolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.improving_and_equal.solver.IESolver._accept

````

`````
