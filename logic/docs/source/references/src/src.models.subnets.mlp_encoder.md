# {py:mod}`src.models.subnets.mlp_encoder`

```{py:module} src.models.subnets.mlp_encoder
```

```{autodoc2-docstring} src.models.subnets.mlp_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLPLayer <src.models.subnets.mlp_encoder.MLPLayer>`
  - ```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPLayer
    :summary:
    ```
* - {py:obj}`MLPEncoder <src.models.subnets.mlp_encoder.MLPEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPEncoder
    :summary:
    ```
````

### API

`````{py:class} MLPLayer(hidden_dim, norm='layer', learn_affine=True, track_norm=False)
:canonical: src.models.subnets.mlp_encoder.MLPLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPLayer.__init__
```

````{py:method} forward(x)
:canonical: src.models.subnets.mlp_encoder.MLPLayer.forward

```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPLayer.forward
```

````

`````

`````{py:class} MLPEncoder(n_layers, feed_forward_hidden, norm='layer', learn_affine=True, track_norm=False, *args, **kwargs)
:canonical: src.models.subnets.mlp_encoder.MLPEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPEncoder.__init__
```

````{py:method} forward(x, graph=None)
:canonical: src.models.subnets.mlp_encoder.MLPEncoder.forward

```{autodoc2-docstring} src.models.subnets.mlp_encoder.MLPEncoder.forward
```

````

`````
