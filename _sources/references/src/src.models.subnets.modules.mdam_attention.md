# {py:mod}`src.models.subnets.modules.mdam_attention`

```{py:module} src.models.subnets.modules.mdam_attention
```

```{autodoc2-docstring} src.models.subnets.modules.mdam_attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiHeadAttentionMDAM <src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM>`
  - ```{autodoc2-docstring} src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM
    :summary:
    ```
````

### API

`````{py:class} MultiHeadAttentionMDAM(embed_dim: int, n_heads: int, last_one: bool = False)
:canonical: src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM.__init__
```

````{py:method} _init_parameters() -> None
:canonical: src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM._init_parameters

```{autodoc2-docstring} src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM._init_parameters
```

````

````{py:method} forward(q: torch.Tensor, h: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
:canonical: src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM.forward

```{autodoc2-docstring} src.models.subnets.modules.mdam_attention.MultiHeadAttentionMDAM.forward
```

````

`````
