# {py:mod}`src.pipeline.rl.core.mvmoe_am`

```{py:module} src.pipeline.rl.core.mvmoe_am
```

```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_am
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MVMoE_AM <src.pipeline.rl.core.mvmoe_am.MVMoE_AM>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_am.MVMoE_AM
    :summary:
    ```
````

### API

````{py:class} MVMoE_AM(policy: typing.Optional[torch.nn.Module] = None, moe_kwargs: typing.Optional[dict] = None, baseline: str = 'rollout', **kwargs)
:canonical: src.pipeline.rl.core.mvmoe_am.MVMoE_AM

Bases: {py:obj}`logic.src.pipeline.rl.core.reinforce.REINFORCE`

```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_am.MVMoE_AM
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_am.MVMoE_AM.__init__
```

````
