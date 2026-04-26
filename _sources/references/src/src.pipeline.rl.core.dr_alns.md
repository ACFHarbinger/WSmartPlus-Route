# {py:mod}`src.pipeline.rl.core.dr_alns`

```{py:module} src.pipeline.rl.core.dr_alns
```

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DRALNSLitModule <src.pipeline.rl.core.dr_alns.DRALNSLitModule>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.core.dr_alns.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.core.dr_alns.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.logger
```

````

`````{py:class} DRALNSLitModule(env: logic.src.envs.dr_alns.DRALNSEnv, agent: logic.src.models.core.dr_alns.DRALNSPPOAgent, lr: float = 0.0003, gamma: float = 0.99, gae_lambda: float = 0.95, clip_epsilon: float = 0.2, value_loss_coef: float = 0.5, entropy_coef: float = 0.01, max_grad_norm: float = 0.5, n_epochs: int = 10, n_steps_per_epoch: int = 2048, batch_size: int = 64, instance_generator: typing.Optional[typing.Any] = None, **kwargs)
:canonical: src.pipeline.rl.core.dr_alns.DRALNSLitModule

Bases: {py:obj}`pytorch_lightning.LightningModule`

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule.__init__
```

````{py:method} reset_buffer() -> None
:canonical: src.pipeline.rl.core.dr_alns.DRALNSLitModule.reset_buffer

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule.reset_buffer
```

````

````{py:method} training_step(batch: typing.Any, batch_idx: int) -> torch.Tensor
:canonical: src.pipeline.rl.core.dr_alns.DRALNSLitModule.training_step

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule.training_step
```

````

````{py:method} collect_experience(n_steps: int)
:canonical: src.pipeline.rl.core.dr_alns.DRALNSLitModule.collect_experience

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule.collect_experience
```

````

````{py:method} _process_buffer() -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor], typing.Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
:canonical: src.pipeline.rl.core.dr_alns.DRALNSLitModule._process_buffer

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule._process_buffer
```

````

````{py:method} configure_optimizers() -> torch.optim.Optimizer
:canonical: src.pipeline.rl.core.dr_alns.DRALNSLitModule.configure_optimizers

```{autodoc2-docstring} src.pipeline.rl.core.dr_alns.DRALNSLitModule.configure_optimizers
```

````

`````
