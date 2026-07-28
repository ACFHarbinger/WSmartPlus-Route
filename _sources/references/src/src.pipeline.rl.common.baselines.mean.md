# {py:mod}`src.pipeline.rl.common.baselines.mean`

```{py:module} src.pipeline.rl.common.baselines.mean
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.mean
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MeanBaseline <src.pipeline.rl.common.baselines.mean.MeanBaseline>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.mean.MeanBaseline
    :summary:
    ```
````

### API

`````{py:class} MeanBaseline(**kwargs)
:canonical: src.pipeline.rl.common.baselines.mean.MeanBaseline

Bases: {py:obj}`src.pipeline.rl.common.baselines.base.Baseline`

```{autodoc2-docstring} src.pipeline.rl.common.baselines.mean.MeanBaseline
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines.mean.MeanBaseline.__init__
```

````{py:method} eval(td: tensordict.TensorDict, reward: torch.Tensor, env: typing.Optional[typing.Any] = None) -> torch.Tensor
:canonical: src.pipeline.rl.common.baselines.mean.MeanBaseline.eval

```{autodoc2-docstring} src.pipeline.rl.common.baselines.mean.MeanBaseline.eval
```

````

`````
