# {py:mod}`src.models.core.moe.moe_temporal_attention_model`

```{py:module} src.models.core.moe.moe_temporal_attention_model
```

```{autodoc2-docstring} src.models.core.moe.moe_temporal_attention_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoETemporalAttentionModel <src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel>`
  - ```{autodoc2-docstring} src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel
    :summary:
    ```
````

### API

`````{py:class} MoETemporalAttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, normalization: str = 'batch', n_heads: int = 8, num_experts: int = 4, k: int = 2, noisy_gating: bool = True, **kwargs: typing.Any)
:canonical: src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel

Bases: {py:obj}`logic.src.models.core.temporal_attention_model.TemporalAttentionModel`

```{autodoc2-docstring} src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel.__init__
```

````{py:method} embed_and_transform(input: typing.Any, edges: typing.Optional[typing.Any] = None) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]
:canonical: src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel.embed_and_transform

```{autodoc2-docstring} src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel.embed_and_transform
```

````

````{py:property} total_experts
:canonical: src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel.total_experts
:type: int

```{autodoc2-docstring} src.models.core.moe.moe_temporal_attention_model.MoETemporalAttentionModel.total_experts
```

````

`````
