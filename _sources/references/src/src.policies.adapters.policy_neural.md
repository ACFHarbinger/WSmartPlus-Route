# {py:mod}`src.policies.adapters.policy_neural`

```{py:module} src.policies.adapters.policy_neural
```

```{autodoc2-docstring} src.policies.adapters.policy_neural
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralPolicy <src.policies.adapters.policy_neural.NeuralPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.policy_neural.NeuralPolicy
    :summary:
    ```
````

### API

`````{py:class} NeuralPolicy
:canonical: src.policies.adapters.policy_neural.NeuralPolicy

Bases: {py:obj}`logic.src.policies.adapters.IPolicy`

```{autodoc2-docstring} src.policies.adapters.policy_neural.NeuralPolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.adapters.policy_neural.NeuralPolicy.execute

```{autodoc2-docstring} src.policies.adapters.policy_neural.NeuralPolicy.execute
```

````

````{py:method} _get_must_go_mask(kwargs: dict, bins: typing.Any, profit_vars: typing.Optional[dict], device: torch.device) -> typing.Optional[torch.Tensor]
:canonical: src.policies.adapters.policy_neural.NeuralPolicy._get_must_go_mask

```{autodoc2-docstring} src.policies.adapters.policy_neural.NeuralPolicy._get_must_go_mask
```

````

````{py:method} _convert_must_go_to_mask(must_go: typing.Any, bins: typing.Any, device: torch.device) -> torch.Tensor
:canonical: src.policies.adapters.policy_neural.NeuralPolicy._convert_must_go_to_mask

```{autodoc2-docstring} src.policies.adapters.policy_neural.NeuralPolicy._convert_must_go_to_mask
```

````

`````
