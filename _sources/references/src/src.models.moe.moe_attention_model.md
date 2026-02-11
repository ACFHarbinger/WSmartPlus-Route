# {py:mod}`src.models.moe.moe_attention_model`

```{py:module} src.models.moe.moe_attention_model
```

```{autodoc2-docstring} src.models.moe.moe_attention_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEAttentionModel <src.models.moe.moe_attention_model.MoEAttentionModel>`
  - ```{autodoc2-docstring} src.models.moe.moe_attention_model.MoEAttentionModel
    :summary:
    ```
````

### API

`````{py:class} MoEAttentionModel(embed_dim, hidden_dim, problem, n_encode_layers=2, n_encode_sublayers=None, n_decode_layers=None, dropout_rate=0.1, normalization='batch', n_heads=8, num_experts=4, k=2, noisy_gating=True, **kwargs)
:canonical: src.models.moe.moe_attention_model.MoEAttentionModel

Bases: {py:obj}`logic.src.models.attention_model.AttentionModel`

```{autodoc2-docstring} src.models.moe.moe_attention_model.MoEAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.moe.moe_attention_model.MoEAttentionModel.__init__
```

````{py:property} total_experts
:canonical: src.models.moe.moe_attention_model.MoEAttentionModel.total_experts

```{autodoc2-docstring} src.models.moe.moe_attention_model.MoEAttentionModel.total_experts
```

````

`````
