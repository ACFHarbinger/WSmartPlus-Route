# {py:mod}`src.models.attention_model.symnco_policy`

```{py:module} src.models.attention_model.symnco_policy
```

```{autodoc2-docstring} src.models.attention_model.symnco_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SymNCOPolicy <src.models.attention_model.symnco_policy.SymNCOPolicy>`
  - ```{autodoc2-docstring} src.models.attention_model.symnco_policy.SymNCOPolicy
    :summary:
    ```
````

### API

`````{py:class} SymNCOPolicy(embed_dim: int = 128, use_projection_head: bool = True, **kwargs)
:canonical: src.models.attention_model.symnco_policy.SymNCOPolicy

Bases: {py:obj}`logic.src.models.attention_model.policy.AttentionModelPolicy`

```{autodoc2-docstring} src.models.attention_model.symnco_policy.SymNCOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.attention_model.symnco_policy.SymNCOPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env, strategy: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> dict
:canonical: src.models.attention_model.symnco_policy.SymNCOPolicy.forward

```{autodoc2-docstring} src.models.attention_model.symnco_policy.SymNCOPolicy.forward
```

````

`````
