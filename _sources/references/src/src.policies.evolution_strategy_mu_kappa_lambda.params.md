# {py:mod}`src.policies.evolution_strategy_mu_kappa_lambda.params`

```{py:module} src.policies.evolution_strategy_mu_kappa_lambda.params
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuKappaLambdaESParams <src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams>`
  - ```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams
    :summary:
    ```
````

### API

`````{py:class} MuKappaLambdaESParams
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams
```

````{py:attribute} mu
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.mu
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.mu
```

````

````{py:attribute} kappa
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.kappa
:type: int
:value: >
   7

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.kappa
```

````

````{py:attribute} lambda_
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.lambda_
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.lambda_
```

````

````{py:attribute} rho
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.rho
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.rho
```

````

````{py:attribute} tau_local
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.tau_local
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.tau_local
```

````

````{py:attribute} tau_global
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.tau_global
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.tau_global
```

````

````{py:attribute} initial_sigma
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.initial_sigma
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.initial_sigma
```

````

````{py:attribute} recombination_type
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.recombination_type
:type: str
:value: >
   'intermediate'

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.recombination_type
```

````

````{py:attribute} max_iterations
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.time_limit
```

````

````{py:attribute} min_sigma
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.min_sigma
:type: float
:value: >
   1e-10

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.min_sigma
```

````

````{py:attribute} max_sigma
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.max_sigma
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.max_sigma
```

````

````{py:attribute} bounds_min
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.bounds_min
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.bounds_min
```

````

````{py:attribute} bounds_max
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.bounds_max
:type: typing.Optional[float]
:value: >
   5.0

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.bounds_max
```

````

````{py:attribute} n_removal
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.n_removal
```

````

````{py:attribute} stagnation_limit
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.stagnation_limit
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.stagnation_limit
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.local_search_iterations
```

````

````{py:method} __post_init__()
:canonical: src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.__post_init__

```{autodoc2-docstring} src.policies.evolution_strategy_mu_kappa_lambda.params.MuKappaLambdaESParams.__post_init__
```

````

`````
