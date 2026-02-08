# {py:mod}`src.models.subnets.modules.matnet_attention`

```{py:module} src.models.subnets.modules.matnet_attention
```

```{autodoc2-docstring} src.models.subnets.modules.matnet_attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MixedScoreMHA <src.models.subnets.modules.matnet_attention.MixedScoreMHA>`
  - ```{autodoc2-docstring} src.models.subnets.modules.matnet_attention.MixedScoreMHA
    :summary:
    ```
````

### API

`````{py:class} MixedScoreMHA(n_heads: int, embed_dim: int, key_dim: typing.Optional[int] = None)
:canonical: src.models.subnets.modules.matnet_attention.MixedScoreMHA

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.matnet_attention.MixedScoreMHA
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.matnet_attention.MixedScoreMHA.__init__
```

````{py:method} init_parameters()
:canonical: src.models.subnets.modules.matnet_attention.MixedScoreMHA.init_parameters

```{autodoc2-docstring} src.models.subnets.modules.matnet_attention.MixedScoreMHA.init_parameters
```

````

````{py:method} forward(row_emb: torch.Tensor, col_emb: torch.Tensor, matrix: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.modules.matnet_attention.MixedScoreMHA.forward

```{autodoc2-docstring} src.models.subnets.modules.matnet_attention.MixedScoreMHA.forward
```

````

`````
