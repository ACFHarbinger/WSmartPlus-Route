# {py:mod}`src.models.subnets.encoders.l2d.encoder`

```{py:module} src.models.subnets.encoders.l2d.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.l2d.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`L2DEncoder <src.models.subnets.encoders.l2d.encoder.L2DEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.l2d.encoder.L2DEncoder
    :summary:
    ```
````

### API

`````{py:class} L2DEncoder(embed_dim: int = 128, num_layers: int = 3, feedforward_hidden: int = 512, **kwargs)
:canonical: src.models.subnets.encoders.l2d.encoder.L2DEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.l2d.encoder.L2DEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.l2d.encoder.L2DEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict) -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.l2d.encoder.L2DEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.l2d.encoder.L2DEncoder.forward
```

````

`````
