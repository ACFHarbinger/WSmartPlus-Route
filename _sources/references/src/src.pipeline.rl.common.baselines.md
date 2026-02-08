# {py:mod}`src.pipeline.rl.common.baselines`

```{py:module} src.pipeline.rl.common.baselines
```

```{autodoc2-docstring} src.pipeline.rl.common.baselines
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.pipeline.rl.common.baselines.rollout
src.pipeline.rl.common.baselines.none
src.pipeline.rl.common.baselines.mean
src.pipeline.rl.common.baselines.critic
src.pipeline.rl.common.baselines.warmup
src.pipeline.rl.common.baselines.pomo
src.pipeline.rl.common.baselines.shared_critic
src.pipeline.rl.common.baselines.exponential
src.pipeline.rl.common.baselines.base
```

## Module Contents

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

* - {py:obj}`__all__ <src.pipeline.rl.common.baselines.__all__>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.__all__
    :summary:
    ```
* - {py:obj}`BASELINE_REGISTRY <src.pipeline.rl.common.baselines.BASELINE_REGISTRY>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.baselines.BASELINE_REGISTRY
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.pipeline.rl.common.baselines.__all__
:value: >
   ['Baseline', 'MeanBaseline', 'NoBaseline', 'CriticBaseline', 'SharedBaseline', 'ExponentialBaseline'...

```{autodoc2-docstring} src.pipeline.rl.common.baselines.__all__
```

````

````{py:data} BASELINE_REGISTRY
:canonical: src.pipeline.rl.common.baselines.BASELINE_REGISTRY
:value: >
   None

```{autodoc2-docstring} src.pipeline.rl.common.baselines.BASELINE_REGISTRY
```

````

````{py:function} get_baseline(name: str, policy: typing.Optional[torch.nn.Module] = None, **kwargs) -> src.pipeline.rl.common.baselines.base.Baseline
:canonical: src.pipeline.rl.common.baselines.get_baseline

```{autodoc2-docstring} src.pipeline.rl.common.baselines.get_baseline
```
````
