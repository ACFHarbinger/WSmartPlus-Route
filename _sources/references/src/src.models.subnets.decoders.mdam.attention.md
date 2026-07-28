# {py:mod}`src.models.subnets.decoders.mdam.attention`

```{py:module} src.models.subnets.decoders.mdam.attention
```

```{autodoc2-docstring} src.models.subnets.decoders.mdam.attention
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_mdam_logits <src.models.subnets.decoders.mdam.attention.compute_mdam_logits>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.mdam.attention.compute_mdam_logits
    :summary:
    ```
````

### API

````{py:function} compute_mdam_logits(query: torch.Tensor, glimpse_K: torch.Tensor, glimpse_V: torch.Tensor, logit_K: torch.Tensor, mask: typing.Optional[torch.Tensor], num_heads: int, project_out: torch.nn.Module, tanh_clipping: float, mask_inner: bool, mask_logits: bool) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.decoders.mdam.attention.compute_mdam_logits

```{autodoc2-docstring} src.models.subnets.decoders.mdam.attention.compute_mdam_logits
```
````
