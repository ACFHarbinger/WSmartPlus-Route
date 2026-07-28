# {py:mod}`src.pipeline.simulations.checkpoints.manager`

```{py:module} src.pipeline.simulations.checkpoints.manager
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.manager
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`checkpoint_manager <src.pipeline.simulations.checkpoints.manager.checkpoint_manager>`
  - ```{autodoc2-docstring} src.pipeline.simulations.checkpoints.manager.checkpoint_manager
    :summary:
    ```
````

### API

````{py:exception} CheckpointError(error_result: dict)
:canonical: src.pipeline.simulations.checkpoints.manager.CheckpointError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.manager.CheckpointError
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.manager.CheckpointError.__init__
```

````

````{py:function} checkpoint_manager(checkpoint, checkpoint_interval: int, state_getter: typing.Callable[[], typing.Any], success_callback: typing.Optional[typing.Callable[[], None]] = None)
:canonical: src.pipeline.simulations.checkpoints.manager.checkpoint_manager

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.manager.checkpoint_manager
```
````
