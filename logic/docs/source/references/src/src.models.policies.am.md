# {py:mod}`src.models.policies.am`

```{py:module} src.models.policies.am
```

```{autodoc2-docstring} src.models.policies.am
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionModelPolicy <src.models.policies.am.AttentionModelPolicy>`
  - ```{autodoc2-docstring} src.models.policies.am.AttentionModelPolicy
    :summary:
    ```
````

### API

`````{py:class} AttentionModelPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_encode_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', **kwargs)
:canonical: src.models.policies.am.AttentionModelPolicy

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.am.AttentionModelPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.am.AttentionModelPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> dict
:canonical: src.models.policies.am.AttentionModelPolicy.forward

```{autodoc2-docstring} src.models.policies.am.AttentionModelPolicy.forward
```

````

`````
