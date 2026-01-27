# {py:mod}`src.pipeline.simulations.checkpoints`

```{py:module} src.pipeline.simulations.checkpoints
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationCheckpoint <src.pipeline.simulations.checkpoints.SimulationCheckpoint>`
  - ```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint
    :summary:
    ```
* - {py:obj}`CheckpointHook <src.pipeline.simulations.checkpoints.CheckpointHook>`
  - ```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`checkpoint_manager <src.pipeline.simulations.checkpoints.checkpoint_manager>`
  - ```{autodoc2-docstring} src.pipeline.simulations.checkpoints.checkpoint_manager
    :summary:
    ```
````

### API

`````{py:class} SimulationCheckpoint(output_dir, checkpoint_dir='temp', policy='', sample_id=0)
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.__init__
```

````{py:method} get_simulation_info()
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint.get_simulation_info

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.get_simulation_info
```

````

````{py:method} get_checkpoint_file(day=None, end_simulation=False)
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint.get_checkpoint_file

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.get_checkpoint_file
```

````

````{py:method} save_state(state, day=0, end_simulation=False)
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint.save_state

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.save_state
```

````

````{py:method} load_state(day=None)
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint.load_state

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.load_state
```

````

````{py:method} find_last_checkpoint_day()
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint.find_last_checkpoint_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.find_last_checkpoint_day
```

````

````{py:method} clear(policy=None, sample_id=None)
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint.clear

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.clear
```

````

````{py:method} delete_checkpoint_day(day)
:canonical: src.pipeline.simulations.checkpoints.SimulationCheckpoint.delete_checkpoint_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.SimulationCheckpoint.delete_checkpoint_day
```

````

`````

`````{py:class} CheckpointHook(checkpoint, checkpoint_interval, state_getter=None)
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.__init__
```

````{py:method} get_current_day()
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.get_current_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.get_current_day
```

````

````{py:method} get_checkpoint_info()
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.get_checkpoint_info

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.get_checkpoint_info
```

````

````{py:method} set_timer(tic)
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.set_timer

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.set_timer
```

````

````{py:method} set_state_getter(state_getter: typing.Callable[[], typing.Any])
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.set_state_getter

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.set_state_getter
```

````

````{py:method} before_day(day)
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.before_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.before_day
```

````

````{py:method} after_day(tic=None, delete_previous=False)
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.after_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.after_day
```

````

````{py:method} on_error(error: Exception) -> typing.Dict
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.on_error

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.on_error
```

````

````{py:method} on_completion(policy=None, sample_id=None)
:canonical: src.pipeline.simulations.checkpoints.CheckpointHook.on_completion

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointHook.on_completion
```

````

`````

````{py:exception} CheckpointError(error_result)
:canonical: src.pipeline.simulations.checkpoints.CheckpointError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointError
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.CheckpointError.__init__
```

````

````{py:function} checkpoint_manager(checkpoint, checkpoint_interval, state_getter, success_callback=None)
:canonical: src.pipeline.simulations.checkpoints.checkpoint_manager

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.checkpoint_manager
```
````
