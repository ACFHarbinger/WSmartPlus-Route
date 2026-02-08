# {py:mod}`src.models.subnets.encoders.matnet.encoder`

```{py:module} src.models.subnets.encoders.matnet.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.matnet.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MatNetEncoder <src.models.subnets.encoders.matnet.encoder.MatNetEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.matnet.encoder.MatNetEncoder
    :summary:
    ```
````

### API

`````{py:class} MatNetEncoder(num_layers: int, embed_dim: int, n_heads: int, feed_forward_hidden: int = 512, normalization: str = 'instance')
:canonical: src.models.subnets.encoders.matnet.encoder.MatNetEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.matnet.encoder.MatNetEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.matnet.encoder.MatNetEncoder.__init__
```

````{py:method} forward(row_emb: torch.Tensor, col_emb: torch.Tensor, matrix: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.matnet.encoder.MatNetEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.matnet.encoder.MatNetEncoder.forward
```

````

`````
