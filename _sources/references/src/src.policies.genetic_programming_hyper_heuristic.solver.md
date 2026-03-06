# {py:mod}`src.policies.genetic_programming_hyper_heuristic.solver`

```{py:module} src.policies.genetic_programming_hyper_heuristic.solver
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPHHSolver <src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver
    :summary:
    ```
````

### API

`````{py:class} GPHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver.solve

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver.solve
```

````

````{py:method} _evaluate_tree(tree: src.policies.genetic_programming_hyper_heuristic.tree.GPNode, init_routes: typing.List[typing.List[int]], n_steps: int) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate_tree

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate_tree
```

````

````{py:method} _apply_tree(tree: src.policies.genetic_programming_hyper_heuristic.tree.GPNode, init_routes: typing.List[typing.List[int]], n_steps: int) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._apply_tree

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._apply_tree
```

````

````{py:method} _build_context(routes: typing.List[typing.List[int]], step: int, total: int) -> typing.Dict[str, float]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_context

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_context
```

````

````{py:method} _tournament(pop: typing.List[src.policies.genetic_programming_hyper_heuristic.tree.GPNode], fitness: typing.List[float]) -> src.policies.genetic_programming_hyper_heuristic.tree.GPNode
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._tournament

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._tournament
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh0

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh1

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh2

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh3

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh4

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._llh4
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_random_solution

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_random_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._cost

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._cost
```

````

`````
