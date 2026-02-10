# {py:mod}`src.models.subnets.encoders.tgc.encoder`

```{py:module} src.models.subnets.encoders.tgc.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TransGraphConvEncoder <src.models.subnets.encoders.tgc.encoder.TransGraphConvEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.tgc.encoder.TransGraphConvEncoder
    :summary:
    ```
````

### API

`````{py:class} TransGraphConvEncoder(n_heads, embed_dim, n_layers, n_sublayers=None, feed_forward_hidden=512, normalization='batch', epsilon_alpha=1e-05, learn_affine=True, track_stats=False, momentum_beta=0.1, locresp_k=1.0, n_groups=3, activation='gelu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=None, dropout_rate=0.1, agg='mean')
:canonical: src.models.subnets.encoders.tgc.encoder.TransGraphConvEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.tgc.encoder.TransGraphConvEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.tgc.encoder.TransGraphConvEncoder.__init__
```

````{py:method} forward(x, edges)
:canonical: src.models.subnets.encoders.tgc.encoder.TransGraphConvEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.tgc.encoder.TransGraphConvEncoder.forward
```

````

`````
