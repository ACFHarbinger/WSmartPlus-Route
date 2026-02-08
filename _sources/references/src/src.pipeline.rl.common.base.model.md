# {py:mod}`src.pipeline.rl.common.base.model`

```{py:module} src.pipeline.rl.common.base.model
```

```{autodoc2-docstring} src.pipeline.rl.common.base.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RL4COLitModule <src.pipeline.rl.common.base.model.RL4COLitModule>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.model.RL4COLitModule
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.base.model.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.base.model.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.base.model.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.base.model.logger
```

````

`````{py:class} RL4COLitModule(env: logic.src.interfaces.env.IEnv, policy: logic.src.interfaces.policy.IPolicy, baseline: typing.Optional[str] = 'rollout', optimizer: str = 'adam', optimizer_kwargs: typing.Optional[dict] = None, lr_scheduler: typing.Optional[str] = None, lr_scheduler_kwargs: typing.Optional[dict] = None, train_data_size: int = 100000, val_data_size: int = 10000, val_dataset_path: typing.Optional[str] = None, batch_size: int = 256, num_workers: int = 4, persistent_workers: bool = True, pin_memory: bool = False, must_go_selector: typing.Optional[logic.src.policies.selection.VectorizedSelector] = None, **kwargs)
:canonical: src.pipeline.rl.common.base.model.RL4COLitModule

Bases: {py:obj}`pytorch_lightning.LightningModule`, {py:obj}`src.pipeline.rl.common.base.data.DataMixin`, {py:obj}`src.pipeline.rl.common.base.optimization.OptimizationMixin`, {py:obj}`src.pipeline.rl.common.base.steps.StepMixin`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.rl.common.base.model.RL4COLitModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.base.model.RL4COLitModule.__init__
```

````{py:method} save_weights(path: str)
:canonical: src.pipeline.rl.common.base.model.RL4COLitModule.save_weights

```{autodoc2-docstring} src.pipeline.rl.common.base.model.RL4COLitModule.save_weights
```

````

````{py:method} _init_baseline()
:canonical: src.pipeline.rl.common.base.model.RL4COLitModule._init_baseline

```{autodoc2-docstring} src.pipeline.rl.common.base.model.RL4COLitModule._init_baseline
```

````

````{py:method} on_train_epoch_start() -> None
:canonical: src.pipeline.rl.common.base.model.RL4COLitModule.on_train_epoch_start

```{autodoc2-docstring} src.pipeline.rl.common.base.model.RL4COLitModule.on_train_epoch_start
```

````

````{py:method} on_train_epoch_end()
:canonical: src.pipeline.rl.common.base.model.RL4COLitModule.on_train_epoch_end

```{autodoc2-docstring} src.pipeline.rl.common.base.model.RL4COLitModule.on_train_epoch_end
```

````

`````
