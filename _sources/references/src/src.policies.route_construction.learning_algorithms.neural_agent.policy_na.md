# {py:mod}`src.policies.route_construction.learning_algorithms.neural_agent.policy_na`

```{py:module} src.policies.route_construction.learning_algorithms.neural_agent.policy_na
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralAgentPolicy <src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy
    :summary:
    ```
````

### API

`````{py:class} NeuralAgentPolicy(config: typing.Optional[typing.Any] = None)
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy.__init__
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy.execute

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy.execute
```

````

````{py:method} _log_params(context: typing.Dict[str, typing.Any], cost_weights: typing.Dict[str, float]) -> None
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._log_params

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._log_params
```

````

````{py:method} _get_mandatory_mask(kwargs: dict, bins: typing.Any, profit_vars: typing.Optional[dict], device: torch.device) -> typing.Optional[torch.Tensor]
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._get_mandatory_mask

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._get_mandatory_mask
```

````

````{py:method} _convert_mandatory_to_mask(mandatory: typing.Any, bins: typing.Any, device: torch.device) -> torch.Tensor
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._convert_mandatory_to_mask

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._convert_mandatory_to_mask
```

````

````{py:method} _run_solver(sub_dist_matrix: typing.Any, sub_wastes: typing.Any, capacity: float, revenue: float, cost_unit: float, values: typing.Any, mandatory_nodes: typing.Any, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgentPolicy._run_solver
```

````

`````
