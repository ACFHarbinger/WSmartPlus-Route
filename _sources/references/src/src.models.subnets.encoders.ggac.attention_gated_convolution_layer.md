# {py:mod}`src.models.subnets.encoders.ggac.attention_gated_convolution_layer`

```{py:module} src.models.subnets.encoders.ggac.attention_gated_convolution_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionGatedConvolutionLayer <src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer
    :summary:
    ```
````

### API

`````{py:class} AttentionGatedConvolutionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, normalization: str, epsilon_alpha: float, learn_affine: bool, track_stats: bool, mbeta: float, lr_k: float, n_groups: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, uniform_range: typing.List[float], gated: bool = True, agg: str = 'sum', bias: bool = True)
:canonical: src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer.__init__
```

````{py:method} forward(h: torch.Tensor, e: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.ggac.attention_gated_convolution_layer.AttentionGatedConvolutionLayer.forward
```

````

`````
