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

`````{py:class} MoEGraphAttentionEncoder(n_heads: int, embed_dim: int, n_layers: int, n_sublayers: typing.Optional[int] = None, feed_forward_hidden: int = 512, normalization: str = 'batch', norm_eps_alpha: float = 1e-05, norm_learn_affine: bool = True, norm_track_stats: bool = False, norm_momentum_beta: float = 0.1, lrnorm_k: float = 1.0, gnorm_groups: int = 3, activation_function: str = 'gelu', af_param: float = 1.0, af_threshold: float = 6.0, af_replacement_value: float = 6.0, af_num_params: int = 3, af_uniform_range: typing.Optional[typing.List[float]] = None, dropout_rate: float = 0.1, agg: typing.Optional[typing.Any] = None, connection_type: str = 'skip', expansion_rate: int = 4, num_experts: int = 4, k: int = 2, noisy_gating: bool = True, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, **kwargs: typing.Any)
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
