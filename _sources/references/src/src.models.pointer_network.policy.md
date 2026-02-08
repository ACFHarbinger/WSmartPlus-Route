# {py:mod}`src.models.pointer_network.policy`

```{py:module} src.models.pointer_network.policy
```

```{autodoc2-docstring} src.models.pointer_network.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerNetworkPolicy <src.models.pointer_network.policy.PointerNetworkPolicy>`
  - ```{autodoc2-docstring} src.models.pointer_network.policy.PointerNetworkPolicy
    :summary:
    ```
````

### API

`````{py:class} PointerNetworkPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 512, **kwargs)
:canonical: src.models.pointer_network.policy.PointerNetworkPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.pointer_network.policy.PointerNetworkPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.pointer_network.policy.PointerNetworkPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.pointer_network.policy.PointerNetworkPolicy.forward

```{autodoc2-docstring} src.models.pointer_network.policy.PointerNetworkPolicy.forward
```

````

`````
