# {py:mod}`src.pipeline.simulations.checkpoints.hooks`

```{py:module} src.pipeline.simulations.checkpoints.hooks
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CheckpointHook <src.pipeline.simulations.checkpoints.hooks.CheckpointHook>`
  - ```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook
    :summary:
    ```
````

### API

`````{py:class} CheckpointHook(checkpoint, checkpoint_interval: int, state_getter: typing.Optional[typing.Callable[[], typing.Any]] = None)
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.__init__
```

````{py:method} get_current_day() -> int
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.get_current_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.get_current_day
```

````

````{py:method} get_checkpoint_info() -> dict
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.get_checkpoint_info

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.get_checkpoint_info
```

````

````{py:method} set_timer(tic: float) -> None
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.set_timer

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.set_timer
```

````

````{py:method} set_state_getter(state_getter: typing.Callable[[], typing.Any]) -> None
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.set_state_getter

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.set_state_getter
```

````

````{py:method} before_day(day: int) -> None
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.before_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.before_day
```

````

````{py:method} after_day(tic: typing.Optional[float] = None, delete_previous: bool = False) -> None
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.after_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.after_day
```

````

````{py:method} on_error(error: Exception) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.on_error

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.on_error
```

````

````{py:method} on_completion(policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> None
:canonical: src.pipeline.simulations.checkpoints.hooks.CheckpointHook.on_completion

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.hooks.CheckpointHook.on_completion
```

````

`````
