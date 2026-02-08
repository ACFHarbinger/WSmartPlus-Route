# {py:mod}`src.models.subnets.modules.cross_attention`

```{py:module} src.models.subnets.modules.cross_attention
```

```{autodoc2-docstring} src.models.subnets.modules.cross_attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiHeadCrossAttention <src.models.subnets.modules.cross_attention.MultiHeadCrossAttention>`
  - ```{autodoc2-docstring} src.models.subnets.modules.cross_attention.MultiHeadCrossAttention
    :summary:
    ```
````

### API

`````{py:class} MultiHeadCrossAttention(embed_dim: int, n_heads: int, bias: bool = False, attention_dropout: float = 0.0, store_attn_weights: bool = False, sdpa_fn: typing.Optional[typing.Callable] = None)
:canonical: src.models.subnets.modules.cross_attention.MultiHeadCrossAttention

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.cross_attention.MultiHeadCrossAttention
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.cross_attention.MultiHeadCrossAttention.__init__
```

````{py:method} forward(q_input: torch.Tensor, kv_input: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.modules.cross_attention.MultiHeadCrossAttention.forward

```{autodoc2-docstring} src.models.subnets.modules.cross_attention.MultiHeadCrossAttention.forward
```

````

````{py:method} _manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: typing.Optional[torch.Tensor]) -> torch.Tensor
:canonical: src.models.subnets.modules.cross_attention.MultiHeadCrossAttention._manual_attention

```{autodoc2-docstring} src.models.subnets.modules.cross_attention.MultiHeadCrossAttention._manual_attention
```

````

`````
