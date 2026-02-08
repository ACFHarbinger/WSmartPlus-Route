# {py:mod}`src.pipeline.rl.common.baselines.base`

```{py:module} src.pipeline.rl.common.baselines.base
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Baseline <src.pipeline.rl.common.baselines.base.Baseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.baselines.base.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.baselines.base.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.logger
```

````

`````{py:class} Baseline()
:canonical: src.pipeline.rl.common.baselines.base.Baseline

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.base.Baseline.eval
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.eval
```

````

````{py:method} unwrap_batch(batch: typing.Any) -> typing.Tuple[typing.Any, typing.Optional[torch.Tensor]]
:canonical: src.pipeline.rl.common.baselines.base.Baseline.unwrap_batch

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.unwrap_batch
```

````

````{py:method} unwrap_dataset(dataset: typing.Any) -> typing.Any
:canonical: src.pipeline.rl.common.baselines.base.Baseline.unwrap_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.unwrap_dataset
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.common.baselines.base.Baseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.epoch_callback
```

````

````{py:method} wrap_dataset(dataset: typing.Any, policy: typing.Optional[torch.nn.Module] = None, env: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.pipeline.rl.common.baselines.base.Baseline.wrap_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.wrap_dataset
```

````

````{py:method} setup(policy: torch.nn.Module)
:canonical: src.pipeline.rl.common.baselines.base.Baseline.setup

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.setup
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.common.baselines.base.Baseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.common.baselines.base.Baseline.get_learnable_parameters
```

````

`````
