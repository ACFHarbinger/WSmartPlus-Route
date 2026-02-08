# {py:mod}`src.models.subnets.encoders.mlp.encoder`

```{py:module} src.models.subnets.encoders.mlp.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.mlp.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLPEncoder <src.models.subnets.encoders.mlp.encoder.MLPEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.mlp.encoder.MLPEncoder
    :summary:
    ```
````

### API

`````{py:class} MLPEncoder(n_layers, feed_forward_hidden, norm='layer', learn_affine=True, track_norm=False, *args, **kwargs)
:canonical: src.models.subnets.encoders.mlp.encoder.MLPEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.mlp.encoder.MLPEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.mlp.encoder.MLPEncoder.__init__
```

````{py:method} forward(x, graph=None)
:canonical: src.models.subnets.encoders.mlp.encoder.MLPEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.mlp.encoder.MLPEncoder.forward
```

````

`````
