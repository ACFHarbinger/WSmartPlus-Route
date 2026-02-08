# {py:mod}`src.pipeline.rl.common.baselines.critic`

```{py:module} src.pipeline.rl.common.baselines.critic
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.critic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CriticBaseline <src.pipeline.rl.common.baselines.critic.CriticBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.critic.CriticBaseline
    :summary:
    ```
````

### API

`````{py:class} CriticBaseline(critic: typing.Optional[torch.nn.Module] = None, **kwargs)
:canonical: src.pipeline.rl.common.baselines.critic.CriticBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.base.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.critic.CriticBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.critic.CriticBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.critic.CriticBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.critic.CriticBaseline.eval
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.common.baselines.critic.CriticBaseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.common.baselines.critic.CriticBaseline.get_learnable_parameters
```

````

````{py:method} state_dict(*args, **kwargs)
:canonical: src.pipeline.rl.common.baselines.critic.CriticBaseline.state_dict

```{autodoc2-docstring} src.pipeline.rl.common.baselines.critic.CriticBaseline.state_dict
```

````

`````
