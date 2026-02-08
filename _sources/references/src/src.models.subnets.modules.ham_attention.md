# {py:mod}`src.models.subnets.modules.ham_attention`

```{py:module} src.models.subnets.modules.ham_attention
```

```{autodoc2-docstring} src.models.subnets.modules.ham_attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HeterogeneousAttentionLayer <src.models.subnets.modules.ham_attention.HeterogeneousAttentionLayer>`
  - ```{autodoc2-docstring} src.models.subnets.modules.ham_attention.HeterogeneousAttentionLayer
    :summary:
    ```
````

### API

`````{py:class} HeterogeneousAttentionLayer(node_types: typing.List[str], edge_types: typing.List[typing.Tuple[str, str, str]], embed_dim: int, num_heads: int, feedforward_hidden: int = 512, normalization: str = 'instance')
:canonical: src.models.subnets.modules.ham_attention.HeterogeneousAttentionLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.ham_attention.HeterogeneousAttentionLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.ham_attention.HeterogeneousAttentionLayer.__init__
```

````{py:method} forward(x_dict: typing.Dict[str, torch.Tensor], mask_dict: typing.Optional[typing.Dict[str, torch.Tensor]] = None) -> typing.Dict[str, torch.Tensor]
:canonical: src.models.subnets.modules.ham_attention.HeterogeneousAttentionLayer.forward

```{autodoc2-docstring} src.models.subnets.modules.ham_attention.HeterogeneousAttentionLayer.forward
```

````

`````
