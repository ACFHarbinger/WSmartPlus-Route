# {py:mod}`src.models.subnets.decoders.gat.graph_decoder`

```{py:module} src.models.subnets.decoders.gat.graph_decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.graph_decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphAttentionDecoder <src.models.subnets.decoders.gat.graph_decoder.GraphAttentionDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.gat.graph_decoder.GraphAttentionDecoder
    :summary:
    ```
````

### API

`````{py:class} GraphAttentionDecoder(n_heads: int, embed_dim: int, n_layers: int, feed_forward_hidden: int = 512, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, dropout_rate: float = 0.1, **kwargs)
:canonical: src.models.subnets.decoders.gat.graph_decoder.GraphAttentionDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.gat.graph_decoder.GraphAttentionDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.gat.graph_decoder.GraphAttentionDecoder.__init__
```

````{py:method} forward(q, h=None, mask=None)
:canonical: src.models.subnets.decoders.gat.graph_decoder.GraphAttentionDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.gat.graph_decoder.GraphAttentionDecoder.forward
```

````

`````
