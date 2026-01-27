# {py:mod}`src.models.policies.symnco`

```{py:module} src.models.policies.symnco
```

```{autodoc2-docstring} src.models.policies.symnco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SymNCOPolicy <src.models.policies.symnco.SymNCOPolicy>`
  - ```{autodoc2-docstring} src.models.policies.symnco.SymNCOPolicy
    :summary:
    ```
````

### API

`````{py:class} SymNCOPolicy(embed_dim: int = 128, use_projection_head: bool = True, **kwargs)
:canonical: src.models.policies.symnco.SymNCOPolicy

Bases: {py:obj}`logic.src.models.policies.am.AttentionModelPolicy`

```{autodoc2-docstring} src.models.policies.symnco.SymNCOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.symnco.SymNCOPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env, decode_type: str = 'sampling', num_starts: int = 1, actions: typing.Optional[torch.Tensor] = None, **kwargs) -> dict
:canonical: src.models.policies.symnco.SymNCOPolicy.forward

```{autodoc2-docstring} src.models.policies.symnco.SymNCOPolicy.forward
```

````

`````
