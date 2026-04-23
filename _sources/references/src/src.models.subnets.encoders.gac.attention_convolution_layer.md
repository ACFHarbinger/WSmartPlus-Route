# {py:mod}`src.models.subnets.encoders.gac.attention_convolution_layer`

```{py:module} src.models.subnets.encoders.gac.attention_convolution_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionConvolutionLayer <src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer
    :summary:
    ```
````

### API

`````{py:class} AttentionConvolutionLayer(n_heads: int, embed_dim: int, feed_forward_hidden: int, agg: str, normalization: str, epsilon_alpha: float, learn_affine: bool, track_stats: bool, mbeta: float, lr_k: float, n_groups: int, activation: str, af_param: float, threshold: float, replacement_value: float, n_params: int, uniform_range: typing.List[float])
:canonical: src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer.__init__
```

````{py:method} forward(h: torch.Tensor, edges: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.gac.attention_convolution_layer.AttentionConvolutionLayer.forward
```

````

`````
