# {py:mod}`src.models.moe_model`

```{py:module} src.models.moe_model
```

```{autodoc2-docstring} src.models.moe_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEAttentionModel <src.models.moe_model.MoEAttentionModel>`
  - ```{autodoc2-docstring} src.models.moe_model.MoEAttentionModel
    :summary:
    ```
* - {py:obj}`MoETemporalAttentionModel <src.models.moe_model.MoETemporalAttentionModel>`
  - ```{autodoc2-docstring} src.models.moe_model.MoETemporalAttentionModel
    :summary:
    ```
````

### API

````{py:class} MoEAttentionModel(embedding_dim, hidden_dim, problem, n_encode_layers=2, n_encode_sublayers=None, n_decode_layers=None, dropout_rate=0.1, normalization='batch', n_heads=8, num_experts=4, k=2, noisy_gating=True, **kwargs)
:canonical: src.models.moe_model.MoEAttentionModel

Bases: {py:obj}`src.models.attention_model.AttentionModel`

```{autodoc2-docstring} src.models.moe_model.MoEAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.moe_model.MoEAttentionModel.__init__
```

````

````{py:class} MoETemporalAttentionModel(embedding_dim, hidden_dim, problem, n_encode_layers=2, n_encode_sublayers=None, n_decode_layers=None, dropout_rate=0.1, normalization='batch', n_heads=8, num_experts=4, k=2, noisy_gating=True, **kwargs)
:canonical: src.models.moe_model.MoETemporalAttentionModel

Bases: {py:obj}`src.models.temporal_am.TemporalAttentionModel`

```{autodoc2-docstring} src.models.moe_model.MoETemporalAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.moe_model.MoETemporalAttentionModel.__init__
```

````
