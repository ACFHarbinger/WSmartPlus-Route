# {py:mod}`src.models.attention_model.ham_policy`

```{py:module} src.models.attention_model.ham_policy
```

```{autodoc2-docstring} src.models.attention_model.ham_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HAMPolicy <src.models.attention_model.ham_policy.HAMPolicy>`
  - ```{autodoc2-docstring} src.models.attention_model.ham_policy.HAMPolicy
    :summary:
    ```
````

### API

````{py:class} HAMPolicy(embed_dim: int = 128, num_encoder_layers: int = 3, num_heads: int = 8, normalization: str = 'layer', feedforward_hidden: int = 512, env_name: str = 'pdp', **kwargs)
:canonical: src.models.attention_model.ham_policy.HAMPolicy

Bases: {py:obj}`logic.src.models.attention_model.policy.AttentionModelPolicy`

```{autodoc2-docstring} src.models.attention_model.ham_policy.HAMPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.ham_policy.HAMPolicy.__init__
```

````
