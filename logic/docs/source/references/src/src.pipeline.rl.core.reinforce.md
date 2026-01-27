# {py:mod}`src.pipeline.rl.core.reinforce`

```{py:module} src.pipeline.rl.core.reinforce
```

```{autodoc2-docstring} src.pipeline.rl.core.reinforce
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`REINFORCE <src.pipeline.rl.core.reinforce.REINFORCE>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.reinforce.REINFORCE
    :summary:
    ```
````

### API

`````{py:class} REINFORCE(entropy_weight: float = 0.0, max_grad_norm: float = 1.0, lr_critic: float = 0.0001, **kwargs)
:canonical: src.pipeline.rl.core.reinforce.REINFORCE

Bases: {py:obj}`logic.src.pipeline.rl.common.base.RL4COLitModule`

```{autodoc2-docstring} src.pipeline.rl.core.reinforce.REINFORCE
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.reinforce.REINFORCE.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.reinforce.REINFORCE.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.reinforce.REINFORCE.calculate_loss
```

````

````{py:method} on_before_optimizer_step(optimizer)
:canonical: src.pipeline.rl.core.reinforce.REINFORCE.on_before_optimizer_step

```{autodoc2-docstring} src.pipeline.rl.core.reinforce.REINFORCE.on_before_optimizer_step
```

````

`````
