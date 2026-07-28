# {py:mod}`src.models.subnets.modules.moe_feed_forward`

```{py:module} src.models.subnets.modules.moe_feed_forward
```

```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEFeedForward <src.models.subnets.modules.moe_feed_forward.MoEFeedForward>`
  - ```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward.MoEFeedForward
    :summary:
    ```
````

### API

`````{py:class} MoEFeedForward(embed_dim: int, feed_forward_hidden: int, activation: str, af_param: typing.Optional[float], threshold: float, replacement_value: float, n_params: int, dist_range: typing.Optional[typing.Tuple[float, float]], bias: bool = True, num_experts: int = 4, k: int = 2, noisy_gating: bool = True)
:canonical: src.models.subnets.modules.moe_feed_forward.MoEFeedForward

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward.MoEFeedForward
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward.MoEFeedForward.__init__
```

````{py:method} forward(h: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.modules.moe_feed_forward.MoEFeedForward.forward

```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward.MoEFeedForward.forward
```

````

`````
