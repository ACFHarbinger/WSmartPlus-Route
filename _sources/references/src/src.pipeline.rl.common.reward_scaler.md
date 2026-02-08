# {py:mod}`src.pipeline.rl.common.reward_scaler`

```{py:module} src.pipeline.rl.common.reward_scaler
```

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RewardScaler <src.pipeline.rl.common.reward_scaler.RewardScaler>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler
    :summary:
    ```
````

### API

`````{py:class} RewardScaler(scale: typing.Literal[norm, src.pipeline.rl.common.reward_scaler.RewardScaler.__init__.scale, none] = 'norm', running_momentum: float = 0.0, eps: float = 1e-08)
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.__init__
```

````{py:property} mean
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.mean
:type: float

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.mean
```

````

````{py:property} variance
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.variance
:type: float

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.variance
```

````

````{py:property} std
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.std
:type: float

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.std
```

````

````{py:method} update(scores: torch.Tensor) -> None
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.update

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.update
```

````

````{py:method} _update_welford(scores: torch.Tensor) -> None
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler._update_welford

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler._update_welford
```

````

````{py:method} _update_ema(scores: torch.Tensor) -> None
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler._update_ema

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler._update_ema
```

````

````{py:method} __call__(scores: torch.Tensor, update: bool = True) -> torch.Tensor
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.__call__

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.__call__
```

````

````{py:method} reset() -> None
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.reset

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.reset
```

````

````{py:method} state_dict() -> dict
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.state_dict

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.state_dict
```

````

````{py:method} load_state_dict(state: dict) -> None
:canonical: src.pipeline.rl.common.reward_scaler.RewardScaler.load_state_dict

```{autodoc2-docstring} src.pipeline.rl.common.reward_scaler.RewardScaler.load_state_dict
```

````

`````
