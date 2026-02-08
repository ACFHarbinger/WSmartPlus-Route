# {py:mod}`src.pipeline.features.eval.evaluators.multi_start`

```{py:module} src.pipeline.features.eval.evaluators.multi_start
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.multi_start
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiStartEval <src.pipeline.features.eval.evaluators.multi_start.MultiStartEval>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluators.multi_start.MultiStartEval
    :summary:
    ```
````

### API

`````{py:class} MultiStartEval(env: typing.Any, num_starts: typing.Optional[int] = None, progress: bool = True, **kwargs)
:canonical: src.pipeline.features.eval.evaluators.multi_start.MultiStartEval

Bases: {py:obj}`logic.src.pipeline.features.eval.eval_base.EvalBase`

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.multi_start.MultiStartEval
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.multi_start.MultiStartEval.__init__
```

````{py:method} __call__(policy: typing.Any, data_loader: torch.utils.data.DataLoader, **kwargs) -> dict
:canonical: src.pipeline.features.eval.evaluators.multi_start.MultiStartEval.__call__

````

`````
