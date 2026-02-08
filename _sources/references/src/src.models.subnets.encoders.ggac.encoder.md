# {py:mod}`src.models.subnets.encoders.ggac.encoder`

```{py:module} src.models.subnets.encoders.ggac.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GatedGraphAttConvEncoder <src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder
    :summary:
    ```
````

### API

`````{py:class} GatedGraphAttConvEncoder(n_heads, embed_dim, n_layers, n_sublayers=None, feed_forward_hidden=512, normalization='batch', epsilon_alpha=1e-05, learn_affine=True, track_stats=False, momentum_beta=0.1, locresp_k=1.0, n_groups=3, activation='gelu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=[0.125, 1 / 3], dropout_rate=0.1, agg='sum')
:canonical: src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder.__init__
```

````{py:method} forward(x, edges=None, dist=None)
:canonical: src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.ggac.encoder.GatedGraphAttConvEncoder.forward
```

````

`````
