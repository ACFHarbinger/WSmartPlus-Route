# {py:mod}`src.pipeline.simulations.checkpoints.persistence`

```{py:module} src.pipeline.simulations.checkpoints.persistence
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationCheckpoint <src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint>`
  - ```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint
    :summary:
    ```
````

### API

`````{py:class} SimulationCheckpoint(output_dir: str, checkpoint_dir: str = 'temp', policy: str = '', sample_id: int = 0)
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.__init__
```

````{py:method} get_simulation_info() -> dict
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.get_simulation_info

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.get_simulation_info
```

````

````{py:method} get_checkpoint_file(day: typing.Optional[int] = None, end_simulation: bool = False) -> str
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.get_checkpoint_file

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.get_checkpoint_file
```

````

````{py:method} save_state(state: typing.Any, day: int = 0, end_simulation: bool = False) -> None
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.save_state

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.save_state
```

````

````{py:method} load_state(day: typing.Optional[int] = None) -> typing.Tuple[typing.Optional[typing.Any], int]
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.load_state

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.load_state
```

````

````{py:method} find_last_checkpoint_day() -> int
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.find_last_checkpoint_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.find_last_checkpoint_day
```

````

````{py:method} clear(policy: typing.Optional[str] = None, sample_id: typing.Optional[int] = None) -> int
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.clear

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.clear
```

````

````{py:method} delete_checkpoint_day(day: int) -> bool
:canonical: src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.delete_checkpoint_day

```{autodoc2-docstring} src.pipeline.simulations.checkpoints.persistence.SimulationCheckpoint.delete_checkpoint_day
```

````

`````
