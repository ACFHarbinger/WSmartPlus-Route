# {py:mod}`src.models.subnets.encoders.moe.moe_multi_head_attention_layer`

```{py:module} src.models.subnets.encoders.moe.moe_multi_head_attention_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEMultiHeadAttentionLayer <src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer
    :summary:
    ```
````

### API

`````{py:class} MoEMultiHeadAttentionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, normalization: str, epsilon_alpha: float, learn_affine: bool, track_stats: bool, mbeta: float, lr_k: float, n_groups: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, uniform_range: typing.List[float], connection_type: str = 'skip', expansion_rate: int = 4, num_experts: int = 4, k: int = 2, noisy_gating: bool = True)
:canonical: src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer

Bases: {py:obj}`logic.src.models.subnets.encoders.common.MultiHeadAttentionLayerBase`

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer.__init__
```

````{py:method} _create_feed_forward(embed_dim: int, feed_forward_hidden: int, activation_config: logic.src.configs.models.activation_function.ActivationConfig, connection_type: str, expansion_rate: int) -> torch.nn.Module
:canonical: src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer._create_feed_forward

```{autodoc2-docstring} src.models.subnets.encoders.moe.moe_multi_head_attention_layer.MoEMultiHeadAttentionLayer._create_feed_forward
```

````

`````
