# {py:mod}`src.pipeline.features.eval.evaluators.augmentation`

```{py:module} src.pipeline.features.eval.evaluators.augmentation
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.augmentation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AugmentationEval <src.pipeline.features.eval.evaluators.augmentation.AugmentationEval>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluators.augmentation.AugmentationEval
    :summary:
    ```
````

### API

`````{py:class} AugmentationEval(env: typing.Any, num_augment: int = 8, augment_fn: str = 'dihedral8', progress: bool = True, **kwargs)
:canonical: src.pipeline.features.eval.evaluators.augmentation.AugmentationEval

Bases: {py:obj}`logic.src.pipeline.features.eval.eval_base.EvalBase`

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.augmentation.AugmentationEval
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.augmentation.AugmentationEval.__init__
```

````{py:method} __call__(policy: typing.Any, data_loader: torch.utils.data.DataLoader, return_results: bool = False, **kwargs) -> dict
:canonical: src.pipeline.features.eval.evaluators.augmentation.AugmentationEval.__call__

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.augmentation.AugmentationEval.__call__
```

````

`````
