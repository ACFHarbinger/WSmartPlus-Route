# {py:mod}`src.models.policies.pointer`

```{py:module} src.models.policies.pointer
```

```{autodoc2-docstring} src.models.policies.pointer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerNetworkPolicy <src.models.policies.pointer.PointerNetworkPolicy>`
  - ```{autodoc2-docstring} src.models.policies.pointer.PointerNetworkPolicy
    :summary:
    ```
````

### API

`````{py:class} PointerNetworkPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 512, **kwargs)
:canonical: src.models.policies.pointer.PointerNetworkPolicy

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.pointer.PointerNetworkPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.pointer.PointerNetworkPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> dict
:canonical: src.models.policies.pointer.PointerNetworkPolicy.forward

```{autodoc2-docstring} src.models.policies.pointer.PointerNetworkPolicy.forward
```

````

`````
