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

`````{py:class} MoEFeedForward(embed_dim, feed_forward_hidden, activation, af_param, threshold, replacement_value, n_params, dist_range, bias=True, num_experts=4, k=2, noisy_gating=True)
:canonical: src.models.subnets.modules.moe_feed_forward.MoEFeedForward

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward.MoEFeedForward
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward.MoEFeedForward.__init__
```

````{py:method} forward(h, mask=None)
:canonical: src.models.subnets.modules.moe_feed_forward.MoEFeedForward.forward

```{autodoc2-docstring} src.models.subnets.modules.moe_feed_forward.MoEFeedForward.forward
```

````

`````
