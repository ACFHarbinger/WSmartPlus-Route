# {py:mod}`src.models.core.moe.policy`

```{py:module} src.models.core.moe.policy
```

```{autodoc2-docstring} src.models.core.moe.policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEPolicy <src.models.core.moe.policy.MoEPolicy>`
  - ```{autodoc2-docstring} src.models.core.moe.policy.MoEPolicy
    :summary:
    ```
````

### API

````{py:class} MoEPolicy(env_name: str, embed_dim: int = 128, hidden_dim: int = 128, n_encode_layers: int = 3, n_heads: int = 8, normalization: str = 'batch', num_experts: int = 4, k: int = 2, noisy_gating: bool = True, **kwargs)
:canonical: src.models.core.moe.policy.MoEPolicy

Bases: {py:obj}`logic.src.models.core.attention_model.AttentionModelPolicy`

```{autodoc2-docstring} src.models.core.moe.policy.MoEPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.moe.policy.MoEPolicy.__init__
```

````
