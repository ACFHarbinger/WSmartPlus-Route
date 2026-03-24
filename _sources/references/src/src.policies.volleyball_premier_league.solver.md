# {py:mod}`src.policies.volleyball_premier_league.solver`

```{py:module} src.policies.volleyball_premier_league.solver
```

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VPLSolver <src.policies.volleyball_premier_league.solver.VPLSolver>`
  - ```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver
    :summary:
    ```
````

### API

`````{py:class} VPLSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.volleyball_premier_league.params.VPLParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver.solve

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver.solve
```

````

````{py:method} _initialize_population(pop_size: int) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver._initialize_population

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver._initialize_population
```

````

````{py:method} _substitution_phase(active_teams: typing.List[typing.List[typing.List[int]]], passive_teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver._substitution_phase

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver._substitution_phase
```

````

````{py:method} _coaching_phase(active_teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver._coaching_phase

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver._coaching_phase
```

````

````{py:method} _learn_from_elite(current_team: typing.List[typing.List[int]], top1: typing.List[typing.List[int]], top2: typing.List[typing.List[int]], top3: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver._learn_from_elite

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver._learn_from_elite
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver._evaluate

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.volleyball_premier_league.solver.VPLSolver._cost

```{autodoc2-docstring} src.policies.volleyball_premier_league.solver.VPLSolver._cost
```

````

`````
