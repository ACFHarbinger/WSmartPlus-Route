# {py:mod}`src.models.subnets.encoders.ham.encoder`

```{py:module} src.models.subnets.encoders.ham.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.ham.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HAMEncoder <src.models.subnets.encoders.ham.encoder.HAMEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.ham.encoder.HAMEncoder
    :summary:
    ```
````

### API

`````{py:class} HAMEncoder(embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3, feedforward_hidden: int = 512, normalization: str = 'instance')
:canonical: src.models.subnets.encoders.ham.encoder.HAMEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.ham.encoder.HAMEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.ham.encoder.HAMEncoder.__init__
```

````{py:method} forward(embeddings: torch.Tensor, *args, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.ham.encoder.HAMEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.ham.encoder.HAMEncoder.forward
```

````

`````
