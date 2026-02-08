# {py:mod}`src.models.attention_model.deep_decoder_policy`

```{py:module} src.models.attention_model.deep_decoder_policy
```

```{autodoc2-docstring} src.models.attention_model.deep_decoder_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepDecoderPolicy <src.models.attention_model.deep_decoder_policy.DeepDecoderPolicy>`
  - ```{autodoc2-docstring} src.models.attention_model.deep_decoder_policy.DeepDecoderPolicy
    :summary:
    ```
````

### API

`````{py:class} DeepDecoderPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_encode_layers: int = 3, n_decode_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', dropout_rate: float = 0.1, **kwargs)
:canonical: src.models.attention_model.deep_decoder_policy.DeepDecoderPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.attention_model.deep_decoder_policy.DeepDecoderPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.deep_decoder_policy.DeepDecoderPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.attention_model.deep_decoder_policy.DeepDecoderPolicy.forward

```{autodoc2-docstring} src.models.attention_model.deep_decoder_policy.DeepDecoderPolicy.forward
```

````

`````
