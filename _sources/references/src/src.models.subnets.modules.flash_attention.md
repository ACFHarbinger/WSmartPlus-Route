# {py:mod}`src.models.subnets.modules.flash_attention`

```{py:module} src.models.subnets.modules.flash_attention
```

```{autodoc2-docstring} src.models.subnets.modules.flash_attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiHeadFlashAttention <src.models.subnets.modules.flash_attention.MultiHeadFlashAttention>`
  - ```{autodoc2-docstring} src.models.subnets.modules.flash_attention.MultiHeadFlashAttention
    :summary:
    ```
````

### API

`````{py:class} MultiHeadFlashAttention(n_heads: int, input_dim: int, embed_dim: int, val_dim: typing.Optional[int] = None, key_dim: typing.Optional[int] = None, bias: bool = True, attention_dropout: float = 0.0, store_attn_weights: bool = False, sdpa_fn: typing.Optional[typing.Callable] = None)
:canonical: src.models.subnets.modules.flash_attention.MultiHeadFlashAttention

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.flash_attention.MultiHeadFlashAttention
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.flash_attention.MultiHeadFlashAttention.__init__
```

````{py:attribute} last_attn
:canonical: src.models.subnets.modules.flash_attention.MultiHeadFlashAttention.last_attn
:type: typing.Tuple[typing.Optional[torch.Tensor], typing.Optional[torch.Tensor]]
:value: >
   None

```{autodoc2-docstring} src.models.subnets.modules.flash_attention.MultiHeadFlashAttention.last_attn
```

````

````{py:method} init_parameters() -> None
:canonical: src.models.subnets.modules.flash_attention.MultiHeadFlashAttention.init_parameters

```{autodoc2-docstring} src.models.subnets.modules.flash_attention.MultiHeadFlashAttention.init_parameters
```

````

````{py:method} forward(q: torch.Tensor, h: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.modules.flash_attention.MultiHeadFlashAttention.forward

```{autodoc2-docstring} src.models.subnets.modules.flash_attention.MultiHeadFlashAttention.forward
```

````

````{py:method} _manual_attention(Q, K, V, mask)
:canonical: src.models.subnets.modules.flash_attention.MultiHeadFlashAttention._manual_attention

```{autodoc2-docstring} src.models.subnets.modules.flash_attention.MultiHeadFlashAttention._manual_attention
```

````

`````
