# {py:mod}`src.pipeline.features.eval.eval_base`

```{py:module} src.pipeline.features.eval.eval_base
```

```{autodoc2-docstring} src.pipeline.features.eval.eval_base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalBase <src.pipeline.features.eval.eval_base.EvalBase>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.eval_base.EvalBase
    :summary:
    ```
````

### API

`````{py:class} EvalBase(env: typing.Any, progress: bool = True, device: str | torch.device = 'cpu', **kwargs)
:canonical: src.pipeline.features.eval.eval_base.EvalBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.features.eval.eval_base.EvalBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.features.eval.eval_base.EvalBase.__init__
```

````{py:method} __call__(policy: typing.Any, data_loader: torch.utils.data.DataLoader, return_results: bool = False, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.features.eval.eval_base.EvalBase.__call__
:abstractmethod:

```{autodoc2-docstring} src.pipeline.features.eval.eval_base.EvalBase.__call__
```

````

`````
