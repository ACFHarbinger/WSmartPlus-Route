# {py:mod}`src.policies.only_improving.solver`

```{py:module} src.policies.only_improving.solver
```

```{autodoc2-docstring} src.policies.only_improving.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OISolver <src.policies.only_improving.solver.OISolver>`
  - ```{autodoc2-docstring} src.policies.only_improving.solver.OISolver
    :summary:
    ```
````

### API

`````{py:class} OISolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.only_improving.solver.OISolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.only_improving.solver.OISolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.only_improving.solver.OISolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.only_improving.solver.OISolver._accept

````

`````
