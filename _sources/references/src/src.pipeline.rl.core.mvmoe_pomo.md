# {py:mod}`src.pipeline.rl.core.mvmoe_pomo`

```{py:module} src.pipeline.rl.core.mvmoe_pomo
```

```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_pomo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MVMoE_POMO <src.pipeline.rl.core.mvmoe_pomo.MVMoE_POMO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_pomo.MVMoE_POMO
    :summary:
    ```
````

### API

````{py:class} MVMoE_POMO(policy: typing.Optional[torch.nn.Module] = None, moe_kwargs: typing.Optional[dict] = None, num_augment: int = 8, augment_fn: typing.Union[str, typing.Callable] = 'dihedral8', first_aug_identity: bool = True, num_starts: typing.Optional[int] = None, **kwargs)
:canonical: src.pipeline.rl.core.mvmoe_pomo.MVMoE_POMO

Bases: {py:obj}`logic.src.pipeline.rl.core.pomo.POMO`

```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_pomo.MVMoE_POMO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.mvmoe_pomo.MVMoE_POMO.__init__
```

````
