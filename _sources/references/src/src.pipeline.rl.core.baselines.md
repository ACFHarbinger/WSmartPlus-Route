# {py:mod}`src.pipeline.rl.core.baselines`

```{py:module} src.pipeline.rl.core.baselines
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Baseline <src.pipeline.rl.core.baselines.Baseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline
    :summary:
    ```
* - {py:obj}`NoBaseline <src.pipeline.rl.core.baselines.NoBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.NoBaseline
    :summary:
    ```
* - {py:obj}`ExponentialBaseline <src.pipeline.rl.core.baselines.ExponentialBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.ExponentialBaseline
    :summary:
    ```
* - {py:obj}`RolloutBaseline <src.pipeline.rl.core.baselines.RolloutBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline
    :summary:
    ```
* - {py:obj}`WarmupBaseline <src.pipeline.rl.core.baselines.WarmupBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.WarmupBaseline
    :summary:
    ```
* - {py:obj}`CriticBaseline <src.pipeline.rl.core.baselines.CriticBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.CriticBaseline
    :summary:
    ```
* - {py:obj}`POMOBaseline <src.pipeline.rl.core.baselines.POMOBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.POMOBaseline
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_baseline <src.pipeline.rl.core.baselines.get_baseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.get_baseline
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BASELINE_REGISTRY <src.pipeline.rl.core.baselines.BASELINE_REGISTRY>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.baselines.BASELINE_REGISTRY
    :summary:
    ```
````

### API

`````{py:class} Baseline()
:canonical: src.pipeline.rl.core.baselines.Baseline

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.Baseline.eval
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.eval
```

````

````{py:method} unwrap_batch(batch: typing.Any) -> typing.Tuple[typing.Any, typing.Optional[torch.Tensor]]
:canonical: src.pipeline.rl.core.baselines.Baseline.unwrap_batch

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.unwrap_batch
```

````

````{py:method} unwrap_dataset(dataset: typing.Any) -> typing.Any
:canonical: src.pipeline.rl.core.baselines.Baseline.unwrap_dataset

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.unwrap_dataset
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.core.baselines.Baseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.epoch_callback
```

````

````{py:method} wrap_dataset(dataset: typing.Any, policy: typing.Optional[torch.nn.Module] = None, env: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.pipeline.rl.core.baselines.Baseline.wrap_dataset

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.wrap_dataset
```

````

````{py:method} setup(policy: torch.nn.Module)
:canonical: src.pipeline.rl.core.baselines.Baseline.setup

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.setup
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.core.baselines.Baseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.core.baselines.Baseline.get_learnable_parameters
```

````

`````

`````{py:class} NoBaseline(**kwargs)
:canonical: src.pipeline.rl.core.baselines.NoBaseline

Bases: {py:obj}`src.pipeline.rl.core.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.core.baselines.NoBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines.NoBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.NoBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.core.baselines.NoBaseline.eval
```

````

`````

`````{py:class} ExponentialBaseline(beta: float = 0.8, exp_beta: typing.Optional[float] = None, **kwargs)
:canonical: src.pipeline.rl.core.baselines.ExponentialBaseline

Bases: {py:obj}`src.pipeline.rl.core.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.core.baselines.ExponentialBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines.ExponentialBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.ExponentialBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.core.baselines.ExponentialBaseline.eval
```

````

`````

`````{py:class} RolloutBaseline(policy: typing.Optional[torch.nn.Module] = None, update_every: int = 1, bl_alpha: float = 0.05, **kwargs)
:canonical: src.pipeline.rl.core.baselines.RolloutBaseline

Bases: {py:obj}`src.pipeline.rl.core.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline.__init__
```

````{py:method} setup(policy: torch.nn.Module)
:canonical: src.pipeline.rl.core.baselines.RolloutBaseline.setup

```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline.setup
```

````

````{py:method} _rollout(policy: torch.nn.Module, td_or_dataset: typing.Any, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.RolloutBaseline._rollout

```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline._rollout
```

````

````{py:method} wrap_dataset(dataset: typing.Any, policy: typing.Optional[torch.nn.Module] = None, env: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.pipeline.rl.core.baselines.RolloutBaseline.wrap_dataset

```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline.wrap_dataset
```

````

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.RolloutBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline.eval
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.core.baselines.RolloutBaseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.core.baselines.RolloutBaseline.epoch_callback
```

````

`````

`````{py:class} WarmupBaseline(baseline: src.pipeline.rl.core.baselines.Baseline, warmup_epochs: int = 1, bl_warmup_epochs: typing.Optional[int] = None, beta: float = 0.8, exp_beta: typing.Optional[float] = None, **kwargs)
:canonical: src.pipeline.rl.core.baselines.WarmupBaseline

Bases: {py:obj}`src.pipeline.rl.core.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.core.baselines.WarmupBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines.WarmupBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.WarmupBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.core.baselines.WarmupBaseline.eval
```

````

````{py:method} unwrap_batch(batch: typing.Any) -> typing.Tuple[typing.Any, typing.Optional[torch.Tensor]]
:canonical: src.pipeline.rl.core.baselines.WarmupBaseline.unwrap_batch

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.core.baselines.WarmupBaseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.core.baselines.WarmupBaseline.epoch_callback
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.core.baselines.WarmupBaseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.core.baselines.WarmupBaseline.get_learnable_parameters
```

````

`````

`````{py:class} CriticBaseline(critic: typing.Optional[torch.nn.Module] = None, **kwargs)
:canonical: src.pipeline.rl.core.baselines.CriticBaseline

Bases: {py:obj}`src.pipeline.rl.core.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.core.baselines.CriticBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines.CriticBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.CriticBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.core.baselines.CriticBaseline.eval
```

````

````{py:method} get_learnable_parameters() -> list
:canonical: src.pipeline.rl.core.baselines.CriticBaseline.get_learnable_parameters

```{autodoc2-docstring} src.pipeline.rl.core.baselines.CriticBaseline.get_learnable_parameters
```

````

````{py:method} state_dict(*args, **kwargs)
:canonical: src.pipeline.rl.core.baselines.CriticBaseline.state_dict

```{autodoc2-docstring} src.pipeline.rl.core.baselines.CriticBaseline.state_dict
```

````

`````

`````{py:class} POMOBaseline(**kwargs)
:canonical: src.pipeline.rl.core.baselines.POMOBaseline

Bases: {py:obj}`src.pipeline.rl.core.baselines.Baseline`

```{autodoc2-docstring} src.pipeline.rl.core.baselines.POMOBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.baselines.POMOBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.core.baselines.POMOBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.core.baselines.POMOBaseline.eval
```

````

`````

````{py:data} BASELINE_REGISTRY
:canonical: src.pipeline.rl.core.baselines.BASELINE_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.pipeline.rl.core.baselines.BASELINE_REGISTRY
```

````

````{py:function} get_baseline(name: str, policy: typing.Optional[torch.nn.Module] = None, **kwargs) -> src.pipeline.rl.core.baselines.Baseline
:canonical: src.pipeline.rl.core.baselines.get_baseline

```{autodoc2-docstring} src.pipeline.rl.core.baselines.get_baseline
```
````
