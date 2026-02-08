# {py:mod}`src.pipeline.features.eval.evaluators.greedy`

```{py:module} src.pipeline.features.eval.evaluators.greedy
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.greedy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GreedyEval <src.pipeline.features.eval.evaluators.greedy.GreedyEval>`
  - ```{autodoc2-docstring} src.pipeline.features.eval.evaluators.greedy.GreedyEval
    :summary:
    ```
````

### API

`````{py:class} GreedyEval(env: typing.Any, progress: bool = True, device: str | torch.device = 'cpu', **kwargs)
:canonical: src.pipeline.features.eval.evaluators.greedy.GreedyEval

Bases: {py:obj}`logic.src.pipeline.features.eval.eval_base.EvalBase`

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.greedy.GreedyEval
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.features.eval.evaluators.greedy.GreedyEval.__init__
```

````{py:method} __call__(policy: typing.Any, data_loader: torch.utils.data.DataLoader, return_results: bool = False, **kwargs) -> dict
:canonical: src.pipeline.features.eval.evaluators.greedy.GreedyEval.__call__

````

`````
