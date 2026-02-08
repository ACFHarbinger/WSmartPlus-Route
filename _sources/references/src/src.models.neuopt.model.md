# {py:mod}`src.models.neuopt.model`

```{py:module} src.models.neuopt.model
```

```{autodoc2-docstring} src.models.neuopt.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOpt <src.models.neuopt.model.NeuOpt>`
  - ```{autodoc2-docstring} src.models.neuopt.model.NeuOpt
    :summary:
    ```
````

### API

`````{py:class} NeuOpt(env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, policy: typing.Optional[src.models.neuopt.policy.NeuOptPolicy] = None, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3, **policy_kwargs)
:canonical: src.models.neuopt.model.NeuOpt

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.neuopt.model.NeuOpt
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.neuopt.model.NeuOpt.__init__
```

````{py:method} forward(td: typing.Any, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', **kwargs)
:canonical: src.models.neuopt.model.NeuOpt.forward

```{autodoc2-docstring} src.models.neuopt.model.NeuOpt.forward
```

````

`````
