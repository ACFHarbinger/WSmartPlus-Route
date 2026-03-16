# {py:mod}`src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl`

```{py:module} src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MuCommaLambdaESPolicy <src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy>`
  - ```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy
    :summary:
    ```
````

### API

`````{py:class} MuCommaLambdaESPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.MuCommaLambdaESConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy._get_config_key

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy._run_solver

```{autodoc2-docstring} src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl.MuCommaLambdaESPolicy._run_solver
```

````

`````
