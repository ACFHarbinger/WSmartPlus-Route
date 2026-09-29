# {py:mod}`src.pipeline.rl.common.baselines.rollout`

```{py:module} src.pipeline.rl.common.baselines.rollout
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RolloutBaseline <src.pipeline.rl.common.baselines.rollout.RolloutBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.baselines.rollout.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.baselines.rollout.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.logger
```

````

`````{py:class} RolloutBaseline(policy: typing.Optional[torch.nn.Module] = None, update_every: int = 1, bl_alpha: float = 0.05, **kwargs)
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.base.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.__init__
```

````{py:method} setup(policy: torch.nn.Module)
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.setup

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.setup
```

````

````{py:method} configure_comparison(env: typing.Any, sample_size: int, seed: int = 0) -> None
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.configure_comparison

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.configure_comparison
```

````

````{py:method} get_extra_state() -> typing.Dict[str, int]
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.get_extra_state

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.get_extra_state
```

````

````{py:method} set_extra_state(state: typing.Dict[str, int]) -> None
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.set_extra_state

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.set_extra_state
```

````

````{py:method} _load_from_state_dict(state_dict, prefix, *args, **kwargs)
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline._load_from_state_dict

````

````{py:method} _generate_comparison(generation: int) -> torch.utils.data.Dataset
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline._generate_comparison

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline._generate_comparison
```

````

````{py:method} train(mode: bool = True) -> src.pipeline.rl.common.baselines.rollout.RolloutBaseline
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.train

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.train
```

````

````{py:method} _rollout(policy: torch.nn.Module, td_or_dataset: typing.Any, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline._rollout

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline._rollout
```

````

````{py:method} _invoke_greedy(policy: torch.nn.Module, td: typing.Any, env: typing.Optional[typing.Any]) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline._invoke_greedy
:staticmethod:

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline._invoke_greedy
```

````

````{py:method} _rollout_dataset(policy: torch.nn.Module, dataset: typing.Any, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline._rollout_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline._rollout_dataset
```

````

````{py:method} _rollout_batch(policy: torch.nn.Module, td: typing.Any, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline._rollout_batch

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline._rollout_batch
```

````

````{py:method} wrap_dataset(dataset: typing.Any, policy: typing.Optional[torch.nn.Module] = None, env: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.wrap_dataset

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.wrap_dataset
```

````

````{py:method} eval(td: typing.Any, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.eval
```

````

````{py:method} epoch_callback(policy: torch.nn.Module, epoch: int, val_dataset: typing.Optional[typing.Any] = None, env: typing.Optional[typing.Any] = None)
:canonical: src.pipeline.rl.common.baselines.rollout.RolloutBaseline.epoch_callback

```{autodoc2-docstring} src.pipeline.rl.common.baselines.rollout.RolloutBaseline.epoch_callback
```

````

`````
