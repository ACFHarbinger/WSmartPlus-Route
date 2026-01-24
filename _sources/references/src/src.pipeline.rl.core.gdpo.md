# {py:mod}`src.pipeline.rl.core.gdpo`

```{py:module} src.pipeline.rl.core.gdpo
```

```{autodoc2-docstring} src.pipeline.rl.core.gdpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GDPO <src.pipeline.rl.core.gdpo.GDPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.gdpo.GDPO
    :summary:
    ```
````

### API

`````{py:class} GDPO(gdpo_objective_keys: typing.List[str], gdpo_objective_weights: typing.Optional[typing.List[float]] = None, gdpo_conditional_key: typing.Optional[str] = None, gdpo_renormalize: bool = True, **kwargs)
:canonical: src.pipeline.rl.core.gdpo.GDPO

Bases: {py:obj}`logic.src.pipeline.rl.core.reinforce.REINFORCE`

```{autodoc2-docstring} src.pipeline.rl.core.gdpo.GDPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.gdpo.GDPO.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.gdpo.GDPO.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.gdpo.GDPO.calculate_loss
```

````

`````
