# {py:mod}`src.models.core.moe.moe_attention_model`

```{py:module} src.models.core.moe.moe_attention_model
```

```{autodoc2-docstring} src.models.core.moe.moe_attention_model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEAttentionModel <src.models.core.moe.moe_attention_model.MoEAttentionModel>`
  - ```{autodoc2-docstring} src.models.core.moe.moe_attention_model.MoEAttentionModel
    :summary:
    ```
````

### API

`````{py:class} MoEAttentionModel(embed_dim: int, hidden_dim: int, problem: typing.Any, n_encode_layers: int = 2, n_encode_sublayers: typing.Optional[int] = None, n_decode_layers: typing.Optional[int] = None, dropout_rate: float = 0.1, normalization: str = 'batch', n_heads: int = 8, num_experts: int = 4, k: int = 2, noisy_gating: bool = True, **kwargs: typing.Any)
:canonical: src.models.core.moe.moe_attention_model.MoEAttentionModel

Bases: {py:obj}`logic.src.models.core.attention_model.AttentionModel`

```{autodoc2-docstring} src.models.core.moe.moe_attention_model.MoEAttentionModel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.moe.moe_attention_model.MoEAttentionModel.__init__
```

````{py:property} total_experts
:canonical: src.models.core.moe.moe_attention_model.MoEAttentionModel.total_experts
:type: int

```{autodoc2-docstring} src.models.core.moe.moe_attention_model.MoEAttentionModel.total_experts
```

````

`````
