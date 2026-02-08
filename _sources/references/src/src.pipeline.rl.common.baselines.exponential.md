# {py:mod}`src.pipeline.rl.common.baselines.exponential`

```{py:module} src.pipeline.rl.common.baselines.exponential
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.exponential
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExponentialBaseline <src.pipeline.rl.common.baselines.exponential.ExponentialBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.exponential.ExponentialBaseline
    :summary:
    ```
````

### API

`````{py:class} ExponentialBaseline(beta: float = 0.8, exp_beta: typing.Optional[float] = None, **kwargs)
:canonical: src.pipeline.rl.common.baselines.exponential.ExponentialBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.base.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.exponential.ExponentialBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.exponential.ExponentialBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.exponential.ExponentialBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.exponential.ExponentialBaseline.eval
```

````

`````
