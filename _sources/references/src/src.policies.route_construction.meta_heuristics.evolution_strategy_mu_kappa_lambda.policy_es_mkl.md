# {py:mod}`src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl`

```{py:module} src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuKappaLambdaESPolicy <src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy
    :summary:
    ```
````

### API

`````{py:class} MuKappaLambdaESPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.MuKappaLambdaESConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.policy_es_mkl.MuKappaLambdaESPolicy._run_solver
```

````

`````
