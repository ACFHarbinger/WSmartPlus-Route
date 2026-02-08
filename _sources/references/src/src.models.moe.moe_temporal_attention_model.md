# {py:mod}`src.models.moe.moe_temporal_attention_model`

```{py:module} src.models.moe.moe_temporal_attention_model
```

```{autodoc2-docstring} src.models.moe.moe_temporal_attention_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoETemporalAttentionModel <src.models.moe.moe_temporal_attention_model.MoETemporalAttentionModel>`
  - ```{autodoc2-docstring} src.models.moe.moe_temporal_attention_model.MoETemporalAttentionModel
    :summary:
    ```
````

### API

`````{py:class} MoETemporalAttentionModel(embed_dim, hidden_dim, problem, n_encode_layers=2, n_encode_sublayers=None, n_decode_layers=None, dropout_rate=0.1, normalization='batch', n_heads=8, num_experts=4, k=2, noisy_gating=True, **kwargs)
:canonical: src.models.moe.moe_temporal_attention_model.MoETemporalAttentionModel

Bases: {py:obj}`logic.src.models.temporal_attention_model.TemporalAttentionModel`

```{autodoc2-docstring} src.models.moe.moe_temporal_attention_model.MoETemporalAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.moe.moe_temporal_attention_model.MoETemporalAttentionModel.__init__
```

````{py:method} embed_and_transform(input, edges=None)
:canonical: src.models.moe.moe_temporal_attention_model.MoETemporalAttentionModel.embed_and_transform

```{autodoc2-docstring} src.models.moe.moe_temporal_attention_model.MoETemporalAttentionModel.embed_and_transform
```

````

`````
