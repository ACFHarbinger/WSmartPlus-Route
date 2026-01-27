# {py:mod}`src.pipeline.rl.core.imitation`

```{py:module} src.pipeline.rl.core.imitation
```

```{autodoc2-docstring} src.pipeline.rl.core.imitation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImitationLearning <src.pipeline.rl.core.imitation.ImitationLearning>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.imitation.ImitationLearning
    :summary:
    ```
````

### API

`````{py:class} ImitationLearning(expert_policy: typing.Any = None, expert_name: str = 'hgs', **kwargs)
:canonical: src.pipeline.rl.core.imitation.ImitationLearning

Bases: {py:obj}`logic.src.pipeline.rl.common.base.RL4COLitModule`

```{autodoc2-docstring} src.pipeline.rl.core.imitation.ImitationLearning
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.imitation.ImitationLearning.__init__
```

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Any = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.imitation.ImitationLearning.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.imitation.ImitationLearning.calculate_loss
```

````

`````
