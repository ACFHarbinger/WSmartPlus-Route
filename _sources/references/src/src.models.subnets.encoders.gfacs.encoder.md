# {py:mod}`src.models.subnets.encoders.gfacs.encoder`

```{py:module} src.models.subnets.encoders.gfacs.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.gfacs.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GFACSEncoder <src.models.subnets.encoders.gfacs.encoder.GFACSEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.gfacs.encoder.GFACSEncoder
    :summary:
    ```
````

### API

`````{py:class} GFACSEncoder(embed_dim: int = 128, num_layers: int = 3, num_heads: int = 8, feedforward_dim: int = 512, dropout: float = 0.0, input_dim: int = 2, **kwargs)
:canonical: src.models.subnets.encoders.gfacs.encoder.GFACSEncoder

Bases: {py:obj}`logic.src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder`

```{autodoc2-docstring} src.models.subnets.encoders.gfacs.encoder.GFACSEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.gfacs.encoder.GFACSEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, return_embeddings: bool = False, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: src.models.subnets.encoders.gfacs.encoder.GFACSEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.gfacs.encoder.GFACSEncoder.forward
```

````

`````
