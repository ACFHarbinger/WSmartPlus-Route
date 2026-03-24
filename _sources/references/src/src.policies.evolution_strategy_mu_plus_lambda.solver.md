# {py:mod}`src.policies.evolution_strategy_mu_plus_lambda.solver`

```{py:module} src.policies.evolution_strategy_mu_plus_lambda.solver
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuPlusLambdaESSolver <src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver>`
  - ```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver
    :summary:
    ```
````

### API

`````{py:class} MuPlusLambdaESSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.evolution_strategy_mu_plus_lambda.params.MuPlusLambdaESParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver.solve

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver.solve
```

````

````{py:method} _initialize_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._initialize_solution

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._initialize_solution
```

````

````{py:method} _recombine_and_mutate(parent1: typing.List[typing.List[int]], parent2: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._recombine_and_mutate

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._recombine_and_mutate
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._evaluate

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._cost

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.solver.MuPlusLambdaESSolver._cost
```

````

`````
