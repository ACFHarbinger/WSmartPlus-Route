# {py:mod}`src.models.subnets.encoders.deepaco.encoder`

```{py:module} src.models.subnets.encoders.deepaco.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.deepaco.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepACOEncoder <src.models.subnets.encoders.deepaco.encoder.DeepACOEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.deepaco.encoder.DeepACOEncoder
    :summary:
    ```
````

### API

`````{py:class} DeepACOEncoder(embed_dim: int = 128, num_layers: int = 3, num_heads: int = 8, feedforward_dim: int = 512, dropout: float = 0.0, input_dim: int = 2, **kwargs)
:canonical: src.models.subnets.encoders.deepaco.encoder.DeepACOEncoder

Bases: {py:obj}`logic.src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder`

```{autodoc2-docstring} src.models.subnets.encoders.deepaco.encoder.DeepACOEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.deepaco.encoder.DeepACOEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, return_embeddings: bool = False, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: src.models.subnets.encoders.deepaco.encoder.DeepACOEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.deepaco.encoder.DeepACOEncoder.forward
```

````

`````
