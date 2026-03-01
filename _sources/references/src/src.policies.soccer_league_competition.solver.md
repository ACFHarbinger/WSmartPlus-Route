# {py:mod}`src.policies.soccer_league_competition.solver`

```{py:module} src.policies.soccer_league_competition.solver
```

```{autodoc2-docstring} src.policies.soccer_league_competition.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SLCSolver <src.policies.soccer_league_competition.solver.SLCSolver>`
  - ```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver
    :summary:
    ```
````

### API

`````{py:class} SLCSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.soccer_league_competition.params.SLCParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.soccer_league_competition.solver.SLCSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.soccer_league_competition.solver.SLCSolver.solve

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver.solve
```

````

````{py:method} _new_team() -> typing.List[typing.Tuple[typing.List[typing.List[int]], float]]
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._new_team

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._new_team
```

````

````{py:method} _random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._random_solution

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._random_solution
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._build_random_solution

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._build_random_solution
```

````

````{py:method} _perturb(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._perturb

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._perturb
```

````

````{py:method} _recombine(loser_routes: typing.List[typing.List[int]], winner_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._recombine

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._recombine
```

````

````{py:method} _league_best(teams: typing.List[typing.List[typing.Tuple[typing.List[typing.List[int]], float]]]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._league_best

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._league_best
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._evaluate

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.soccer_league_competition.solver.SLCSolver._cost

```{autodoc2-docstring} src.policies.soccer_league_competition.solver.SLCSolver._cost
```

````

`````
