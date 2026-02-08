# {py:mod}`src.models.subnets.encoders.mdam.encoder`

```{py:module} src.models.subnets.encoders.mdam.encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.mdam.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MDAMGraphAttentionEncoder <src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder
    :summary:
    ```
````

### API

`````{py:class} MDAMGraphAttentionEncoder(num_heads: int, embed_dim: int, num_layers: int, node_dim: typing.Optional[int] = None, normalization: str = 'batch', feed_forward_hidden: int = 512)
:canonical: src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder

Bases: {py:obj}`logic.src.models.common.autoregressive_encoder.AutoregressiveEncoder`

```{autodoc2-docstring} src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, x: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder.forward
```

````

````{py:method} change(attn: torch.Tensor, V: torch.Tensor, h_old: torch.Tensor, mask: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder.change

```{autodoc2-docstring} src.models.subnets.encoders.mdam.encoder.MDAMGraphAttentionEncoder.change
```

````

`````
