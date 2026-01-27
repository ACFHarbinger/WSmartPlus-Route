# {py:mod}`src.models.modules.multi_head_attention`

```{py:module} src.models.modules.multi_head_attention
```

```{autodoc2-docstring} src.models.modules.multi_head_attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiHeadAttention <src.models.modules.multi_head_attention.MultiHeadAttention>`
  - ```{autodoc2-docstring} src.models.modules.multi_head_attention.MultiHeadAttention
    :summary:
    ```
````

### API

`````{py:class} MultiHeadAttention(n_heads: int, input_dim: int, embed_dim: int, val_dim: typing.Optional[int] = None, key_dim: typing.Optional[int] = None)
:canonical: src.models.modules.multi_head_attention.MultiHeadAttention

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.multi_head_attention.MultiHeadAttention
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.multi_head_attention.MultiHeadAttention.__init__
```

````{py:attribute} last_attn
:canonical: src.models.modules.multi_head_attention.MultiHeadAttention.last_attn
:type: typing.Tuple[typing.Optional[torch.Tensor], typing.Optional[torch.Tensor]]
:value: >
   None

```{autodoc2-docstring} src.models.modules.multi_head_attention.MultiHeadAttention.last_attn
```

````

````{py:method} init_parameters() -> None
:canonical: src.models.modules.multi_head_attention.MultiHeadAttention.init_parameters

```{autodoc2-docstring} src.models.modules.multi_head_attention.MultiHeadAttention.init_parameters
```

````

````{py:method} forward(q: torch.Tensor, h: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.modules.multi_head_attention.MultiHeadAttention.forward

```{autodoc2-docstring} src.models.modules.multi_head_attention.MultiHeadAttention.forward
```

````

`````
