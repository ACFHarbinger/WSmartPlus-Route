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

`````{py:class} GraphAttentionDecoder(n_heads, embed_dim, n_layers, feed_forward_hidden=512, normalization='batch', epsilon_alpha=1e-05, learn_affine=True, track_stats=False, momentum_beta=0.1, locresp_k=1.0, n_groups=3, activation='gelu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=[0.125, 1 / 3], dropout_rate=0.1)
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
