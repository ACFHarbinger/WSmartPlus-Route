# {py:mod}`src.pipeline.rl.common.baselines.shared_critic`

```{py:module} src.pipeline.rl.common.baselines.shared_critic
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.shared_critic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SharedBaseline <src.pipeline.rl.common.baselines.shared_critic.SharedBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.shared_critic.SharedBaseline
    :summary:
    ```
````

### API

`````{py:class} SharedBaseline(critic: typing.Optional[torch.nn.Module] = None, **kwargs)
:canonical: src.pipeline.rl.common.baselines.shared_critic.SharedBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.base.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.shared_critic.SharedBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.shared_critic.SharedBaseline.__init__
```

````{py:method} setup(policy: torch.nn.Module)
:canonical: src.pipeline.rl.common.baselines.shared_critic.SharedBaseline.setup

```{autodoc2-docstring} src.pipeline.rl.common.baselines.shared_critic.SharedBaseline.setup
```

````

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.shared_critic.SharedBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.shared_critic.SharedBaseline.eval
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.common.baselines.shared_critic.SharedBaseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.common.baselines.shared_critic.SharedBaseline.get_learnable_parameters
```

````

`````
