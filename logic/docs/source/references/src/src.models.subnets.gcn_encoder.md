# {py:mod}`src.models.subnets.gcn_encoder`

```{py:module} src.models.subnets.gcn_encoder
```

```{autodoc2-docstring} src.models.subnets.gcn_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphConvolutionEncoder <src.models.subnets.gcn_encoder.GraphConvolutionEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.gcn_encoder.GraphConvolutionEncoder
    :summary:
    ```
````

### API

`````{py:class} GraphConvolutionEncoder(n_layers, feed_forward_hidden, agg='sum', norm='layer', learn_affine=True, track_norm=False, gated=True, *args, **kwargs)
:canonical: src.models.subnets.gcn_encoder.GraphConvolutionEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.gcn_encoder.GraphConvolutionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.gcn_encoder.GraphConvolutionEncoder.__init__
```

````{py:method} forward(x, edges)
:canonical: src.models.subnets.gcn_encoder.GraphConvolutionEncoder.forward

```{autodoc2-docstring} src.models.subnets.gcn_encoder.GraphConvolutionEncoder.forward
```

````

`````
