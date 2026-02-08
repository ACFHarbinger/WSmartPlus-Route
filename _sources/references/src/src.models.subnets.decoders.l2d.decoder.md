# {py:mod}`src.models.subnets.decoders.l2d.decoder`

```{py:module} src.models.subnets.decoders.l2d.decoder
```

```{autodoc2-docstring} src.models.subnets.decoders.l2d.decoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`L2DDecoder <src.models.subnets.decoders.l2d.decoder.L2DDecoder>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.l2d.decoder.L2DDecoder
    :summary:
    ```
````

### API

`````{py:class} L2DDecoder(embed_dim: int, temp: float = 1.0, tanh_clipping: float = 10.0)
:canonical: src.models.subnets.decoders.l2d.decoder.L2DDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.decoders.l2d.decoder.L2DDecoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.decoders.l2d.decoder.L2DDecoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, embeddings: typing.Tuple[torch.Tensor, torch.Tensor], env: typing.Optional[typing.Any] = None, decode_type: str = 'sampling', return_pi: bool = False, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.l2d.decoder.L2DDecoder.forward

```{autodoc2-docstring} src.models.subnets.decoders.l2d.decoder.L2DDecoder.forward
```

````

`````
