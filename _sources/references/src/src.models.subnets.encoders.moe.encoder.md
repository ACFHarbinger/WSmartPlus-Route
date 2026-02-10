# {py:mod}`src.models.subnets.encoders.moe.encoder`

```{py:module} src.models.subnets.encoders.moe.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.moe.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEGraphAttentionEncoder <src.models.subnets.encoders.moe.encoder.MoEGraphAttentionEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.moe.encoder.MoEGraphAttentionEncoder
    :summary:
    ```
````

### API

`````{py:class} MoEGraphAttentionEncoder(n_heads, embed_dim, n_layers, n_sublayers=None, feed_forward_hidden=512, normalization='batch', norm_eps_alpha=1e-05, norm_learn_affine=True, norm_track_stats=False, norm_momentum_beta=0.1, lrnorm_k=1.0, gnorm_groups=3, activation_function='gelu', af_param=1.0, af_threshold=6.0, af_replacement_value=6.0, af_num_params=3, af_uniform_range=None, dropout_rate=0.1, agg=None, connection_type='skip', expansion_rate=4, num_experts=4, k=2, noisy_gating=True, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, **kwargs)
:canonical: src.models.subnets.encoders.moe.encoder.MoEGraphAttentionEncoder

Bases: {py:obj}`logic.src.models.subnets.encoders.common.TransformerEncoderBase`

```{autodoc2-docstring} src.models.subnets.encoders.moe.encoder.MoEGraphAttentionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.moe.encoder.MoEGraphAttentionEncoder.__init__
```

````{py:method} _create_layer(layer_idx: int) -> torch.nn.Module
:canonical: src.models.subnets.encoders.moe.encoder.MoEGraphAttentionEncoder._create_layer

```{autodoc2-docstring} src.models.subnets.encoders.moe.encoder.MoEGraphAttentionEncoder._create_layer
```

````

`````
