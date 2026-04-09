# {py:mod}`src.policies.neural_agent.policy_neural`

```{py:module} src.policies.neural_agent.policy_neural
```

```{autodoc2-docstring} src.policies.neural_agent.policy_neural
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralPolicy <src.policies.neural_agent.policy_neural.NeuralPolicy>`
  - ```{autodoc2-docstring} src.policies.neural_agent.policy_neural.NeuralPolicy
    :summary:
    ```
````

### API

`````{py:class} NeuralPolicy(config: typing.Optional[typing.Any] = None)
:canonical: src.policies.neural_agent.policy_neural.NeuralPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.neural_agent.policy_neural.NeuralPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.neural_agent.policy_neural.NeuralPolicy.__init__
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.neural_agent.policy_neural.NeuralPolicy.execute

```{autodoc2-docstring} src.policies.neural_agent.policy_neural.NeuralPolicy.execute
```

````

````{py:method} _log_params(context: typing.Dict[str, typing.Any], cost_weights: typing.Dict[str, float]) -> None
:canonical: src.policies.neural_agent.policy_neural.NeuralPolicy._log_params

```{autodoc2-docstring} src.policies.neural_agent.policy_neural.NeuralPolicy._log_params
```

````

````{py:method} _get_must_go_mask(kwargs: dict, bins: typing.Any, profit_vars: typing.Optional[dict], device: torch.device) -> typing.Optional[torch.Tensor]
:canonical: src.policies.neural_agent.policy_neural.NeuralPolicy._get_must_go_mask

```{autodoc2-docstring} src.policies.neural_agent.policy_neural.NeuralPolicy._get_must_go_mask
```

````

````{py:method} _convert_must_go_to_mask(must_go: typing.Any, bins: typing.Any, device: torch.device) -> torch.Tensor
:canonical: src.policies.neural_agent.policy_neural.NeuralPolicy._convert_must_go_to_mask

```{autodoc2-docstring} src.policies.neural_agent.policy_neural.NeuralPolicy._convert_must_go_to_mask
```

````

`````
