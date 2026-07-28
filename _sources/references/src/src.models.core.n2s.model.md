# {py:mod}`src.models.core.n2s.model`

```{py:module} src.models.core.n2s.model
```

```{autodoc2-docstring} src.models.core.n2s.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`N2S <src.models.core.n2s.model.N2S>`
  - ```{autodoc2-docstring} src.models.core.n2s.model.N2S
    :summary:
    ```
````

### API

`````{py:class} N2S(env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, policy: typing.Optional[src.models.core.n2s.policy.N2SPolicy] = None, embed_dim: int = 128, num_heads: int = 8, k_neighbors: int = 20, **policy_kwargs: typing.Any)
:canonical: src.models.core.n2s.model.N2S

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.core.n2s.model.N2S
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.n2s.model.N2S.__init__
```

````{py:method} forward(td: typing.Any, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, strategy: str = 'greedy', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.n2s.model.N2S.forward

```{autodoc2-docstring} src.models.core.n2s.model.N2S.forward
```

````

`````
