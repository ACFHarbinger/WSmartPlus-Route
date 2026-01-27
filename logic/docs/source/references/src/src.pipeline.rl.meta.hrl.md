# {py:mod}`src.pipeline.rl.meta.hrl`

```{py:module} src.pipeline.rl.meta.hrl
```

```{autodoc2-docstring} src.pipeline.rl.meta.hrl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HRLModule <src.pipeline.rl.meta.hrl.HRLModule>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.hrl.HRLModule
    :summary:
    ```
````

### API

`````{py:class} HRLModule(manager: logic.src.models.gat_lstm_manager.GATLSTManager, worker: logic.src.models.policies.base.ConstructivePolicy, env: logic.src.envs.base.RL4COEnvBase, lr: float = 0.0001, gamma: float = 0.99, clip_eps: float = 0.2, ppo_epochs: int = 4, lambda_mask_aux: float = 0.0, entropy_coef: float = 0.1, **kwargs)
:canonical: src.pipeline.rl.meta.hrl.HRLModule

Bases: {py:obj}`pytorch_lightning.LightningModule`

```{autodoc2-docstring} src.pipeline.rl.meta.hrl.HRLModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.hrl.HRLModule.__init__
```

````{py:method} training_step(batch: tensordict.TensorDict, batch_idx: int)
:canonical: src.pipeline.rl.meta.hrl.HRLModule.training_step

```{autodoc2-docstring} src.pipeline.rl.meta.hrl.HRLModule.training_step
```

````

````{py:method} configure_optimizers()
:canonical: src.pipeline.rl.meta.hrl.HRLModule.configure_optimizers

```{autodoc2-docstring} src.pipeline.rl.meta.hrl.HRLModule.configure_optimizers
```

````

`````
