# {py:mod}`src.models.policies.temporal`

```{py:module} src.models.policies.temporal
```

```{autodoc2-docstring} src.models.policies.temporal
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalAMPolicy <src.models.policies.temporal.TemporalAMPolicy>`
  - ```{autodoc2-docstring} src.models.policies.temporal.TemporalAMPolicy
    :summary:
    ```
````

### API

`````{py:class} TemporalAMPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 512, temporal_horizon: int = 5, predictor_layers: int = 2, **kwargs)
:canonical: src.models.policies.temporal.TemporalAMPolicy

Bases: {py:obj}`logic.src.models.policies.am.AttentionModelPolicy`

```{autodoc2-docstring} src.models.policies.temporal.TemporalAMPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.temporal.TemporalAMPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> dict
:canonical: src.models.policies.temporal.TemporalAMPolicy.forward

```{autodoc2-docstring} src.models.policies.temporal.TemporalAMPolicy.forward
```

````

`````
