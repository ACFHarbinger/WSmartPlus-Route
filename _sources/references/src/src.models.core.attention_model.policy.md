# {py:mod}`src.models.core.attention_model.policy`

```{py:module} src.models.core.attention_model.policy
```

```{autodoc2-docstring} src.models.core.attention_model.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionModelPolicy <src.models.core.attention_model.policy.AttentionModelPolicy>`
  - ```{autodoc2-docstring} src.models.core.attention_model.policy.AttentionModelPolicy
    :summary:
    ```
````

### API

`````{py:class} AttentionModelPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_encode_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', **kwargs: typing.Any)
:canonical: src.models.core.attention_model.policy.AttentionModelPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive.policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.core.attention_model.policy.AttentionModelPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.attention_model.policy.AttentionModelPolicy.__init__
```

````{py:attribute} encoder
:canonical: src.models.core.attention_model.policy.AttentionModelPolicy.encoder
:type: logic.src.models.subnets.encoders.gat.GraphAttentionEncoder
:value: >
   None

```{autodoc2-docstring} src.models.core.attention_model.policy.AttentionModelPolicy.encoder
```

````

````{py:attribute} decoder
:canonical: src.models.core.attention_model.policy.AttentionModelPolicy.decoder
:type: logic.src.models.subnets.decoders.glimpse.decoder.GlimpseDecoder
:value: >
   None

```{autodoc2-docstring} src.models.core.attention_model.policy.AttentionModelPolicy.decoder
```

````

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.base.RL4COEnvBase, strategy: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, start_nodes: typing.Optional[torch.Tensor] = None, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.attention_model.policy.AttentionModelPolicy.forward

```{autodoc2-docstring} src.models.core.attention_model.policy.AttentionModelPolicy.forward
```

````

`````
