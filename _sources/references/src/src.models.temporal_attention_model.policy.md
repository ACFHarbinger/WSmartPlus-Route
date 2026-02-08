# {py:mod}`src.models.temporal_attention_model.policy`

```{py:module} src.models.temporal_attention_model.policy
```

```{autodoc2-docstring} src.models.temporal_attention_model.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalAMPolicy <src.models.temporal_attention_model.policy.TemporalAMPolicy>`
  - ```{autodoc2-docstring} src.models.temporal_attention_model.policy.TemporalAMPolicy
    :summary:
    ```
````

### API

`````{py:class} TemporalAMPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 512, temporal_horizon: int = 5, predictor_layers: int = 2, predictor_type: str = 'gru', **kwargs)
:canonical: src.models.temporal_attention_model.policy.TemporalAMPolicy

Bases: {py:obj}`logic.src.models.attention_model.policy.AttentionModelPolicy`

```{autodoc2-docstring} src.models.temporal_attention_model.policy.TemporalAMPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.temporal_attention_model.policy.TemporalAMPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, strategy: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> dict
:canonical: src.models.temporal_attention_model.policy.TemporalAMPolicy.forward

```{autodoc2-docstring} src.models.temporal_attention_model.policy.TemporalAMPolicy.forward
```

````

`````
