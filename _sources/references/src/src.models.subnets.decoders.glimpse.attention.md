# {py:mod}`src.models.subnets.decoders.glimpse.attention`

```{py:module} src.models.subnets.decoders.glimpse.attention
```

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.attention
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`one_to_many_logits <src.models.subnets.decoders.glimpse.attention.one_to_many_logits>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.glimpse.attention.one_to_many_logits
    :summary:
    ```
* - {py:obj}`make_heads <src.models.subnets.decoders.glimpse.attention.make_heads>`
  - ```{autodoc2-docstring} src.models.subnets.decoders.glimpse.attention.make_heads
    :summary:
    ```
````

### API

````{py:function} one_to_many_logits(query: torch.Tensor, glimpse_K: torch.Tensor, glimpse_V: torch.Tensor, logit_K: torch.Tensor, mask: torch.Tensor, n_heads: int, graph_mask: typing.Optional[torch.Tensor] = None, dist_bias: typing.Optional[torch.Tensor] = None, mask_val: float = -math.inf, tanh_clipping: float = 10.0) -> torch.Tensor
:canonical: src.models.subnets.decoders.glimpse.attention.one_to_many_logits

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.attention.one_to_many_logits
```
````

````{py:function} make_heads(v: torch.Tensor, n_heads: int, num_steps: typing.Optional[int] = None) -> torch.Tensor
:canonical: src.models.subnets.decoders.glimpse.attention.make_heads

```{autodoc2-docstring} src.models.subnets.decoders.glimpse.attention.make_heads
```
````
