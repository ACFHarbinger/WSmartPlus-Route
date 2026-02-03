# {py:mod}`src.pipeline.rl.common.baselines`

```{py:module} src.pipeline.rl.common.baselines
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Baseline <src.pipeline.rl.common.baselines.Baseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline
    :summary:
    ```
* - {py:obj}`NoBaseline <src.pipeline.rl.common.baselines.NoBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.NoBaseline
    :summary:
    ```
* - {py:obj}`ExponentialBaseline <src.pipeline.rl.common.baselines.ExponentialBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.ExponentialBaseline
    :summary:
    ```
* - {py:obj}`RolloutBaseline <src.pipeline.rl.common.baselines.RolloutBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline
    :summary:
    ```
* - {py:obj}`WarmupBaseline <src.pipeline.rl.common.baselines.WarmupBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.WarmupBaseline
    :summary:
    ```
* - {py:obj}`CriticBaseline <src.pipeline.rl.common.baselines.CriticBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.CriticBaseline
    :summary:
    ```
* - {py:obj}`POMOBaseline <src.pipeline.rl.common.baselines.POMOBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.POMOBaseline
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_baseline <src.pipeline.rl.common.baselines.get_baseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.get_baseline
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.baselines.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.logger
    :summary:
    ```
* - {py:obj}`BASELINE_REGISTRY <src.pipeline.rl.common.baselines.BASELINE_REGISTRY>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.BASELINE_REGISTRY
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.baselines.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.baselines.logger
```

````

`````{py:class} Baseline()
:canonical: src.pipeline.rl.common.baselines.Baseline

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.Baseline.eval
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.eval
```

````

````{py:method} unwrap_batch(batch: typing.Any) -> typing.Tuple[typing.Any, typing.Optional[torch.Tensor]]
:canonical: src.pipeline.rl.common.baselines.Baseline.unwrap_batch

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.unwrap_batch
```

````

````{py:method} unwrap_dataset(dataset: typing.Any) -> typing.Any
:canonical: src.pipeline.rl.common.baselines.Baseline.unwrap_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.unwrap_dataset
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.common.baselines.Baseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.epoch_callback
```

````

````{py:method} wrap_dataset(dataset: typing.Any, policy: typing.Optional[torch.nn.Module] = None, env: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.pipeline.rl.common.baselines.Baseline.wrap_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.wrap_dataset
```

````

````{py:method} setup(policy: torch.nn.Module)
:canonical: src.pipeline.rl.common.baselines.Baseline.setup

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.setup
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.common.baselines.Baseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.common.baselines.Baseline.get_learnable_parameters
```

````

`````

`````{py:class} NoBaseline(**kwargs)
:canonical: src.pipeline.rl.common.baselines.NoBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.NoBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.NoBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.NoBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.NoBaseline.eval
```

````

`````

`````{py:class} ExponentialBaseline(beta: float = 0.8, exp_beta: typing.Optional[float] = None, **kwargs)
:canonical: src.pipeline.rl.common.baselines.ExponentialBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.ExponentialBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.ExponentialBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.ExponentialBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.ExponentialBaseline.eval
```

````

`````

`````{py:class} RolloutBaseline(policy: typing.Optional[torch.nn.Module] = None, update_every: int = 1, bl_alpha: float = 0.05, **kwargs)
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline.__init__
```

````{py:method} setup(policy: torch.nn.Module)
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline.setup

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline.setup
```

````

````{py:method} _rollout(policy: torch.nn.Module, td_or_dataset: typing.Any, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline._rollout

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline._rollout
```

````

````{py:method} _rollout_dataset(policy: torch.nn.Module, dataset: typing.Any, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline._rollout_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline._rollout_dataset
```

````

````{py:method} _rollout_batch(policy: torch.nn.Module, td: typing.Any, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline._rollout_batch

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline._rollout_batch
```

````

````{py:method} wrap_dataset(dataset: typing.Any, policy: typing.Optional[torch.nn.Module] = None, env: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline.wrap_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline.wrap_dataset
```

````

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline.eval
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.common.baselines.RolloutBaseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.common.baselines.RolloutBaseline.epoch_callback
```

````

`````

`````{py:class} WarmupBaseline(baseline: src.pipeline.rl.common.baselines.Baseline, warmup_epochs: int = 1, bl_warmup_epochs: typing.Optional[int] = None, beta: float = 0.8, exp_beta: typing.Optional[float] = None, **kwargs)
:canonical: src.pipeline.rl.common.baselines.WarmupBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.WarmupBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.WarmupBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.WarmupBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.WarmupBaseline.eval
```

````

````{py:method} unwrap_batch(batch: typing.Any) -> typing.Tuple[typing.Any, typing.Optional[torch.Tensor]]
:canonical: src.pipeline.rl.common.baselines.WarmupBaseline.unwrap_batch

```{autodoc2-docstring} src.pipeline.rl.common.baselines.WarmupBaseline.unwrap_batch
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.common.baselines.WarmupBaseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.common.baselines.WarmupBaseline.epoch_callback
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.common.baselines.WarmupBaseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.common.baselines.WarmupBaseline.get_learnable_parameters
```

````

`````

`````{py:class} CriticBaseline(critic: typing.Optional[torch.nn.Module] = None, **kwargs)
:canonical: src.pipeline.rl.common.baselines.CriticBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.CriticBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.CriticBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.CriticBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.CriticBaseline.eval
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.common.baselines.CriticBaseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.common.baselines.CriticBaseline.get_learnable_parameters
```

````

````{py:method} state_dict(*args, **kwargs)
:canonical: src.pipeline.rl.common.baselines.CriticBaseline.state_dict

```{autodoc2-docstring} src.pipeline.rl.common.baselines.CriticBaseline.state_dict
```

````

`````

`````{py:class} POMOBaseline(**kwargs)
:canonical: src.pipeline.rl.common.baselines.POMOBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.POMOBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.POMOBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.POMOBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.POMOBaseline.eval
```

````

`````

````{py:data} BASELINE_REGISTRY
:canonical: src.pipeline.rl.common.baselines.BASELINE_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.pipeline.rl.common.baselines.BASELINE_REGISTRY
```

````

````{py:function} get_baseline(name: str, policy: typing.Optional[torch.nn.Module] = None, **kwargs) -> src.pipeline.rl.common.baselines.Baseline
:canonical: src.pipeline.rl.common.baselines.get_baseline

```{autodoc2-docstring} src.pipeline.rl.common.baselines.get_baseline
```
````
