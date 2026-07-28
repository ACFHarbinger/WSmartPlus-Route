# {py:mod}`src.models.core.neuopt.model`

```{py:module} src.models.core.neuopt.model
```

```{autodoc2-docstring} src.models.core.neuopt.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOpt <src.models.core.neuopt.model.NeuOpt>`
  - ```{autodoc2-docstring} src.models.core.neuopt.model.NeuOpt
    :summary:
    ```
````

### API

`````{py:class} NeuOpt(env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, policy: typing.Optional[src.models.core.neuopt.policy.NeuOptPolicy] = None, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3, **policy_kwargs: typing.Any)
:canonical: src.models.core.neuopt.model.NeuOpt

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.core.neuopt.model.NeuOpt
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.neuopt.model.NeuOpt.__init__
```

````{py:method} forward(td: typing.Any, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, strategy: str = 'greedy', **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.models.core.neuopt.model.NeuOpt.forward

```{autodoc2-docstring} src.models.core.neuopt.model.NeuOpt.forward
```

````

`````
