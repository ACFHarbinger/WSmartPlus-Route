# {py:mod}`src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl`

```{py:module} src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuPlusLambdaESPolicy <src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy>`
  - ```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy
    :summary:
    ```
````

### API

`````{py:class} MuPlusLambdaESPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.MuPlusLambdaESConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy._get_config_key

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy._run_solver

```{autodoc2-docstring} src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl.MuPlusLambdaESPolicy._run_solver
```

````

`````
