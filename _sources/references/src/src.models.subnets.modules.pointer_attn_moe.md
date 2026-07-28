# {py:mod}`src.models.subnets.modules.pointer_attn_moe`

```{py:module} src.models.subnets.modules.pointer_attn_moe
```

```{autodoc2-docstring} src.models.subnets.modules.pointer_attn_moe
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointerAttnMoE <src.models.subnets.modules.pointer_attn_moe.PointerAttnMoE>`
  - ```{autodoc2-docstring} src.models.subnets.modules.pointer_attn_moe.PointerAttnMoE
    :summary:
    ```
````

### API

`````{py:class} PointerAttnMoE(embed_dim: int, num_heads: int, num_experts: int = 4, k: int = 2, noisy_gating: bool = True, mask_inner: bool = True, check_nan: bool = True)
:canonical: src.models.subnets.modules.pointer_attn_moe.PointerAttnMoE

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.pointer_attn_moe.PointerAttnMoE
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.pointer_attn_moe.PointerAttnMoE.__init__
```

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, logit_key: torch.Tensor, attn_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.modules.pointer_attn_moe.PointerAttnMoE.forward

```{autodoc2-docstring} src.models.subnets.modules.pointer_attn_moe.PointerAttnMoE.forward
```

````

`````
