# {py:mod}`src.tracking.integrations.simulation`

```{py:module} src.tracking.integrations.simulation
```

```{autodoc2-docstring} src.tracking.integrations.simulation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationRunTracker <src.tracking.integrations.simulation.SimulationRunTracker>`
  - ```{autodoc2-docstring} src.tracking.integrations.simulation.SimulationRunTracker
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_sim_tracker <src.tracking.integrations.simulation.get_sim_tracker>`
  - ```{autodoc2-docstring} src.tracking.integrations.simulation.get_sim_tracker
    :summary:
    ```
````

### API

`````{py:class} SimulationRunTracker(run: logic.src.tracking.core.run.Run, policy_name: str, sample_id: int)
:canonical: src.tracking.integrations.simulation.SimulationRunTracker

```{autodoc2-docstring} src.tracking.integrations.simulation.SimulationRunTracker
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.simulation.SimulationRunTracker.__init__
```

````{py:method} log_day(day: int, metrics: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.integrations.simulation.SimulationRunTracker.log_day

```{autodoc2-docstring} src.tracking.integrations.simulation.SimulationRunTracker.log_day
```

````

````{py:method} log_final(metric_keys: typing.List[str], metric_values: typing.List[typing.Any]) -> None
:canonical: src.tracking.integrations.simulation.SimulationRunTracker.log_final

```{autodoc2-docstring} src.tracking.integrations.simulation.SimulationRunTracker.log_final
```

````

````{py:method} log_failure(error: str) -> None
:canonical: src.tracking.integrations.simulation.SimulationRunTracker.log_failure

```{autodoc2-docstring} src.tracking.integrations.simulation.SimulationRunTracker.log_failure
```

````

`````

````{py:function} get_sim_tracker(policy_name: str, sample_id: int) -> typing.Optional[src.tracking.integrations.simulation.SimulationRunTracker]
:canonical: src.tracking.integrations.simulation.get_sim_tracker

```{autodoc2-docstring} src.tracking.integrations.simulation.get_sim_tracker
```
````
