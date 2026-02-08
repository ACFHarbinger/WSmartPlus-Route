# {py:mod}`src.models.subnets.modules.polynet_attention`

```{py:module} src.models.subnets.modules.polynet_attention
```

```{autodoc2-docstring} src.models.subnets.modules.polynet_attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolyNetAttention <src.models.subnets.modules.polynet_attention.PolyNetAttention>`
  - ```{autodoc2-docstring} src.models.subnets.modules.polynet_attention.PolyNetAttention
    :summary:
    ```
````

### API

`````{py:class} PolyNetAttention(k: int, embed_dim: int, poly_layer_dim: int = 256, num_heads: int = 8, mask_inner: bool = True, out_bias: bool = False, check_nan: bool = True)
:canonical: src.models.subnets.modules.polynet_attention.PolyNetAttention

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.polynet_attention.PolyNetAttention
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.polynet_attention.PolyNetAttention.__init__
```

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, logit_key: torch.Tensor, attn_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.modules.polynet_attention.PolyNetAttention.forward

```{autodoc2-docstring} src.models.subnets.modules.polynet_attention.PolyNetAttention.forward
```

````

````{py:method} _get_strategy_vectors(num_solutions: int, device: torch.device) -> torch.Tensor
:canonical: src.models.subnets.modules.polynet_attention.PolyNetAttention._get_strategy_vectors

```{autodoc2-docstring} src.models.subnets.modules.polynet_attention.PolyNetAttention._get_strategy_vectors
```

````

````{py:method} _inner_mha(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: typing.Optional[torch.Tensor]) -> torch.Tensor
:canonical: src.models.subnets.modules.polynet_attention.PolyNetAttention._inner_mha

```{autodoc2-docstring} src.models.subnets.modules.polynet_attention.PolyNetAttention._inner_mha
```

````

````{py:method} _make_heads(v: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.polynet_attention.PolyNetAttention._make_heads

```{autodoc2-docstring} src.models.subnets.modules.polynet_attention.PolyNetAttention._make_heads
```

````

`````
