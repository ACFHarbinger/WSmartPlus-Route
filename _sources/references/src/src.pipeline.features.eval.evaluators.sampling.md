# {py:mod}`src.pipeline.features.eval.evaluators.sampling`

```{py:module} src.pipeline.features.eval.evaluators.sampling
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.sampling
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SamplingEval <src.pipeline.features.eval.evaluators.sampling.SamplingEval>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluators.sampling.SamplingEval
    :summary:
    ```
````

### API

`````{py:class} SamplingEval(env: typing.Any, samples: int = 1280, progress: bool = True, **kwargs)
:canonical: src.pipeline.features.eval.evaluators.sampling.SamplingEval

Bases: {py:obj}`logic.src.pipeline.features.eval.eval_base.EvalBase`

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.sampling.SamplingEval
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.sampling.SamplingEval.__init__
```

````{py:method} __call__(policy: typing.Any, data_loader: torch.utils.data.DataLoader, return_results: bool = False, **kwargs) -> dict
:canonical: src.pipeline.features.eval.evaluators.sampling.SamplingEval.__call__

````

`````
