# {py:mod}`src.models.core.attention_model.deep_decoder_policy`

```{py:module} src.models.core.attention_model.deep_decoder_policy
```

```{autodoc2-docstring} src.models.core.attention_model.deep_decoder_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepDecoderPolicy <src.models.core.attention_model.deep_decoder_policy.DeepDecoderPolicy>`
  - ```{autodoc2-docstring} src.models.core.attention_model.deep_decoder_policy.DeepDecoderPolicy
    :summary:
    ```
````

### API

`````{py:class} DeepDecoderPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_encode_layers: int = 3, n_decode_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', dropout_rate: float = 0.1, **kwargs: typing.Any)
:canonical: src.models.core.attention_model.deep_decoder_policy.DeepDecoderPolicy

Bases: {py:obj}`logic.src.models.core.attention_model.policy.AttentionModelPolicy`

```{autodoc2-docstring} src.models.core.attention_model.deep_decoder_policy.DeepDecoderPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.attention_model.deep_decoder_policy.DeepDecoderPolicy.__init__
```

````{py:attribute} decoder
:canonical: src.models.core.attention_model.deep_decoder_policy.DeepDecoderPolicy.decoder
:type: logic.src.models.subnets.decoders.gat.DeepGATDecoder
:value: >
   None

```{autodoc2-docstring} src.models.core.attention_model.deep_decoder_policy.DeepDecoderPolicy.decoder
```

````

`````
