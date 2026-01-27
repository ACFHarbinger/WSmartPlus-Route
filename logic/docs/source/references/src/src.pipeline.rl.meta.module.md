# {py:mod}`src.pipeline.rl.meta.module`

```{py:module} src.pipeline.rl.meta.module
```

```{autodoc2-docstring} src.pipeline.rl.meta.module
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MetaRLModule <src.pipeline.rl.meta.module.MetaRLModule>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.module.MetaRLModule
    :summary:
    ```
````

### API

`````{py:class} MetaRLModule(agent: typing.Any, strategy: str = 'rnn', meta_lr: float = 0.001, history_length: int = 10, hidden_size: int = 64, **kwargs)
:canonical: src.pipeline.rl.meta.module.MetaRLModule

Bases: {py:obj}`pytorch_lightning.LightningModule`

```{autodoc2-docstring} src.pipeline.rl.meta.module.MetaRLModule
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.module.MetaRLModule.__init__
```

````{py:method} training_step(batch: tensordict.TensorDict, batch_idx: int)
:canonical: src.pipeline.rl.meta.module.MetaRLModule.training_step

```{autodoc2-docstring} src.pipeline.rl.meta.module.MetaRLModule.training_step
```

````

````{py:method} configure_optimizers()
:canonical: src.pipeline.rl.meta.module.MetaRLModule.configure_optimizers

```{autodoc2-docstring} src.pipeline.rl.meta.module.MetaRLModule.configure_optimizers
```

````

````{py:method} validation_step(batch: tensordict.TensorDict, batch_idx: int)
:canonical: src.pipeline.rl.meta.module.MetaRLModule.validation_step

```{autodoc2-docstring} src.pipeline.rl.meta.module.MetaRLModule.validation_step
```

````

````{py:method} test_step(batch: tensordict.TensorDict, batch_idx: int)
:canonical: src.pipeline.rl.meta.module.MetaRLModule.test_step

```{autodoc2-docstring} src.pipeline.rl.meta.module.MetaRLModule.test_step
```

````

`````
