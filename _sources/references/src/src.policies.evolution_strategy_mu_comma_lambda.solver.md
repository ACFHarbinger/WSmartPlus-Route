# {py:mod}`src.policies.evolution_strategy_mu_comma_lambda.solver`

```{py:module} src.policies.evolution_strategy_mu_comma_lambda.solver
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuCommaLambdaESSolver <src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver>`
  - ```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver
    :summary:
    ```
````

### API

`````{py:class} MuCommaLambdaESSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.evolution_strategy_mu_comma_lambda.params.MuCommaLambdaESParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver.solve

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver.solve
```

````

````{py:method} _initialize_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._initialize_solution

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._initialize_solution
```

````

````{py:method} _recombine_and_mutate(parent1: typing.List[typing.List[int]], parent2: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._recombine_and_mutate

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._recombine_and_mutate
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._evaluate

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._cost

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.solver.MuCommaLambdaESSolver._cost
```

````

`````
