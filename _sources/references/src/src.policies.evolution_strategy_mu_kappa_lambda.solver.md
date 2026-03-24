# {py:mod}`src.policies.evolution_strategy_mu_kappa_lambda.solver`

```{py:module} src.policies.evolution_strategy_mu_kappa_lambda.solver
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuKappaLambdaESSolver <src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver>`
  - ```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver
    :summary:
    ```
````

### API

`````{py:class} MuKappaLambdaESSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver.solve

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver.solve
```

````

````{py:method} _initialize_population() -> typing.List[src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual]
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._initialize_population

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._initialize_population
```

````

````{py:method} _select_parents_for_recombination(parents: typing.List[src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual]) -> typing.List[src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual]
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._select_parents_for_recombination

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._select_parents_for_recombination
```

````

````{py:method} _recombine(parents: typing.List[src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual]) -> src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._recombine

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._recombine
```

````

````{py:method} _mutate(individual: src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual) -> src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._mutate

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._mutate
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._evaluate

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._cost

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._cost
```

````

````{py:method} _update_best(population: typing.List[src.policies.evolution_strategy_mu_kappa_lambda.individual.Individual])
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._update_best

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.solver.MuKappaLambdaESSolver._update_best
```

````

`````
