# {py:mod}`src.policies.sequence_based_selection_hyper_heuristic.solver`

```{py:module} src.policies.sequence_based_selection_hyper_heuristic.solver
```

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_SeqEntry <src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry>`
  - ```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry
    :summary:
    ```
* - {py:obj}`SSHHSolver <src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver>`
  - ```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver
    :summary:
    ```
````

### API

`````{py:class} _SeqEntry
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry
```

````{py:attribute} h_prev
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry.h_prev
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry.h_prev
```

````

````{py:attribute} h_cur
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry.h_cur
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry.h_cur
```

````

````{py:attribute} AS
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry.AS
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver._SeqEntry.AS
```

````

`````

`````{py:class} SSHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.sequence_based_selection_hyper_heuristic.params.SSHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver.solve

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver.solve
```

````

````{py:method} _accept(new_profit: float, candidate_profit: float, best_profit: float, elapsed: float) -> bool
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._accept

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._accept
```

````

````{py:method} _roulette_wheel_row(scores: numpy.ndarray) -> int
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._roulette_wheel_row

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._roulette_wheel_row
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh0

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh1

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh2

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh3

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh4

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh4
```

````

````{py:method} _llh5(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh5

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh5
```

````

````{py:method} _llh6(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh6

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._llh6
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._build_initial_solution

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._evaluate

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._cost

```{autodoc2-docstring} src.policies.sequence_based_selection_hyper_heuristic.solver.SSHHSolver._cost
```

````

`````
