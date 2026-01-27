# {py:mod}`src.pipeline.rl.core.symnco`

```{py:module} src.pipeline.rl.core.symnco
```

```{autodoc2-docstring} src.pipeline.rl.core.symnco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SymNCO <src.pipeline.rl.core.symnco.SymNCO>`
  - ```{autodoc2-docstring} src.pipeline.rl.core.symnco.SymNCO
    :summary:
    ```
````

### API

`````{py:class} SymNCO(alpha: float = 0.2, beta: float = 1.0, **kwargs)
:canonical: src.pipeline.rl.core.symnco.SymNCO

Bases: {py:obj}`logic.src.pipeline.rl.core.pomo.POMO`

```{autodoc2-docstring} src.pipeline.rl.core.symnco.SymNCO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.core.symnco.SymNCO.__init__
```

````{py:method} shared_step(batch: tensordict.TensorDict, batch_idx: int, phase: str) -> dict
:canonical: src.pipeline.rl.core.symnco.SymNCO.shared_step

```{autodoc2-docstring} src.pipeline.rl.core.symnco.SymNCO.shared_step
```

````

`````
