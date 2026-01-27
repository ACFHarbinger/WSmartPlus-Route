# {py:mod}`src.pipeline.rl.core.pomo`

```{py:module} src.pipeline.rl.core.pomo
```

```{autodoc2-docstring} src.pipeline.rl.core.pomo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`POMO <src.pipeline.rl.core.pomo.POMO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.pomo.POMO
    :summary:
    ```
````

### API

`````{py:class} POMO(num_augment: int = 8, augment_fn: typing.Union[str, typing.Callable] = 'dihedral8', first_aug_identity: bool = True, num_starts: typing.Optional[int] = None, **kwargs)
:canonical: src.pipeline.rl.core.pomo.POMO

Bases: {py:obj}`logic.src.pipeline.rl.core.reinforce.REINFORCE`

```{autodoc2-docstring} src.pipeline.rl.core.pomo.POMO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.pomo.POMO.__init__
```

````{py:method} shared_step(batch: tensordict.TensorDict, batch_idx: int, phase: str) -> dict
:canonical: src.pipeline.rl.core.pomo.POMO.shared_step

```{autodoc2-docstring} src.pipeline.rl.core.pomo.POMO.shared_step
```

````

`````
