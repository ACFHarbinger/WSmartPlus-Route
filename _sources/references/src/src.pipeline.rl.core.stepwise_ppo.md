# {py:mod}`src.pipeline.rl.core.stepwise_ppo`

```{py:module} src.pipeline.rl.core.stepwise_ppo
```

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepwisePPO <src.pipeline.rl.core.stepwise_ppo.StepwisePPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO
    :summary:
    ```
````

### API

`````{py:class} StepwisePPO(critic: torch.nn.Module, ppo_epochs: int = 10, eps_clip: float = 0.2, value_loss_weight: float = 0.5, entropy_weight: float = 0.01, max_grad_norm: float = 0.5, mini_batch_size: int | float = 0.25, gamma: float = 0.99, gae_lambda: float = 0.95, **kwargs)
:canonical: src.pipeline.rl.core.stepwise_ppo.StepwisePPO

Bases: {py:obj}`logic.src.pipeline.rl.common.base.RL4COLitModule`

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO.__init__
```

````{py:method} training_step(batch: tensordict.TensorDict, batch_idx: int)
:canonical: src.pipeline.rl.core.stepwise_ppo.StepwisePPO.training_step

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO.training_step
```

````

````{py:method} _process_experiences(experiences: list) -> tensordict.TensorDict
:canonical: src.pipeline.rl.core.stepwise_ppo.StepwisePPO._process_experiences

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO._process_experiences
```

````

````{py:method} _get_mbs(td)
:canonical: src.pipeline.rl.core.stepwise_ppo.StepwisePPO._get_mbs

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO._get_mbs
```

````

````{py:method} calculate_loss(td, out, batch_idx, env=None)
:canonical: src.pipeline.rl.core.stepwise_ppo.StepwisePPO.calculate_loss

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO.calculate_loss
```

````

````{py:method} configure_optimizers()
:canonical: src.pipeline.rl.core.stepwise_ppo.StepwisePPO.configure_optimizers

```{autodoc2-docstring} src.pipeline.rl.core.stepwise_ppo.StepwisePPO.configure_optimizers
```

````

`````
