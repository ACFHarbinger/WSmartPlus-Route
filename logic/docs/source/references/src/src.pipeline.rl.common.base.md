# {py:mod}`src.pipeline.rl.common.base`

```{py:module} src.pipeline.rl.common.base
```

```{autodoc2-docstring} src.pipeline.rl.common.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RL4COLitModule <src.pipeline.rl.common.base.RL4COLitModule>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.base.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.base.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.base.logger
```

````

`````{py:class} RL4COLitModule(env: logic.src.envs.base.RL4COEnvBase, policy: logic.src.models.policies.base.ConstructivePolicy, baseline: typing.Optional[str] = 'rollout', optimizer: str = 'adam', optimizer_kwargs: typing.Optional[dict] = None, lr_scheduler: typing.Optional[str] = None, lr_scheduler_kwargs: typing.Optional[dict] = None, train_data_size: int = 100000, val_data_size: int = 10000, val_dataset_path: typing.Optional[str] = None, batch_size: int = 256, num_workers: int = 4, persistent_workers: bool = True, pin_memory: bool = False, **kwargs)
:canonical: src.pipeline.rl.common.base.RL4COLitModule

Bases: {py:obj}`pytorch_lightning.LightningModule`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.__init__
```

````{py:method} save_weights(path: str)
:canonical: src.pipeline.rl.common.base.RL4COLitModule.save_weights

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.save_weights
```

````

````{py:method} _init_baseline()
:canonical: src.pipeline.rl.common.base.RL4COLitModule._init_baseline

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule._init_baseline
```

````

````{py:method} calculate_loss(td: tensordict.TensorDict, out: dict, batch_idx: int, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.base.RL4COLitModule.calculate_loss
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.calculate_loss
```

````

````{py:method} shared_step(batch: typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]], batch_idx: int, phase: str) -> dict
:canonical: src.pipeline.rl.common.base.RL4COLitModule.shared_step

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.shared_step
```

````

````{py:method} training_step(batch: typing.Any, batch_idx: int) -> torch.Tensor
:canonical: src.pipeline.rl.common.base.RL4COLitModule.training_step

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.training_step
```

````

````{py:method} validation_step(batch: tensordict.TensorDict, batch_idx: int) -> dict
:canonical: src.pipeline.rl.common.base.RL4COLitModule.validation_step

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.validation_step
```

````

````{py:method} test_step(batch: tensordict.TensorDict, batch_idx: int) -> dict
:canonical: src.pipeline.rl.common.base.RL4COLitModule.test_step

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.test_step
```

````

````{py:method} on_train_epoch_start() -> None
:canonical: src.pipeline.rl.common.base.RL4COLitModule.on_train_epoch_start

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.on_train_epoch_start
```

````

````{py:method} on_train_epoch_end()
:canonical: src.pipeline.rl.common.base.RL4COLitModule.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.on_train_epoch_end
```

````

````{py:method} configure_optimizers()
:canonical: src.pipeline.rl.common.base.RL4COLitModule.configure_optimizers

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.configure_optimizers
```

````

````{py:method} setup(stage: str) -> None
:canonical: src.pipeline.rl.common.base.RL4COLitModule.setup

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.setup
```

````

````{py:method} train_dataloader() -> torch.utils.data.DataLoader
:canonical: src.pipeline.rl.common.base.RL4COLitModule.train_dataloader

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.train_dataloader
```

````

````{py:method} val_dataloader() -> torch.utils.data.DataLoader
:canonical: src.pipeline.rl.common.base.RL4COLitModule.val_dataloader

```{autodoc2-docstring} src.pipeline.rl.common.base.RL4COLitModule.val_dataloader
```

````

`````
