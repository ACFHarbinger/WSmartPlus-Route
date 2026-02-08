# {py:mod}`src.models.subnets.encoders.matnet.matnet_encoder_layer`

```{py:module} src.models.subnets.encoders.matnet.matnet_encoder_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.matnet.matnet_encoder_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MatNetEncoderLayer <src.models.subnets.encoders.matnet.matnet_encoder_layer.MatNetEncoderLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.matnet.matnet_encoder_layer.MatNetEncoderLayer
    :summary:
    ```
````

### API

`````{py:class} MatNetEncoderLayer(embed_dim: int, n_heads: int, feed_forward_hidden: int = 512, normalization: str = 'instance')
:canonical: src.models.subnets.encoders.matnet.matnet_encoder_layer.MatNetEncoderLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.matnet.matnet_encoder_layer.MatNetEncoderLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.matnet.matnet_encoder_layer.MatNetEncoderLayer.__init__
```

````{py:method} forward(row_emb: torch.Tensor, col_emb: torch.Tensor, matrix: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.matnet.matnet_encoder_layer.MatNetEncoderLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.matnet.matnet_encoder_layer.MatNetEncoderLayer.forward
```

````

`````
