# {py:mod}`src.pipeline.features.eval.evaluators.combined`

```{py:module} src.pipeline.features.eval.evaluators.combined
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.combined
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiStartAugmentEval <src.pipeline.features.eval.evaluators.combined.MultiStartAugmentEval>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluators.combined.MultiStartAugmentEval
    :summary:
    ```
````

### API

`````{py:class} MultiStartAugmentEval(env: typing.Any, num_augment: int = 8, num_starts: typing.Optional[int] = None, progress: bool = True, **kwargs)
:canonical: src.pipeline.features.eval.evaluators.combined.MultiStartAugmentEval

Bases: {py:obj}`logic.src.pipeline.features.eval.eval_base.EvalBase`

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.combined.MultiStartAugmentEval
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.combined.MultiStartAugmentEval.__init__
```

````{py:method} __call__(policy: typing.Any, data_loader: torch.utils.data.DataLoader, **kwargs) -> dict
:canonical: src.pipeline.features.eval.evaluators.combined.MultiStartAugmentEval.__call__

````

`````
