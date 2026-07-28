# {py:mod}`src.pipeline.callbacks.simulation.simulation_display`

```{py:module} src.pipeline.callbacks.simulation.simulation_display
```

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationDisplayCallback <src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback
    :summary:
    ```
````

### API

`````{py:class} SimulationDisplayCallback(policies: typing.List[str], n_samples: int, total_days: int, chart_metric: str = 'profit', refresh_rate: int = 2, theme: str = 'dark')
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback.__init__
```

````{py:method} _init_layout() -> None
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._init_layout

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._init_layout
```

````

````{py:method} start() -> None
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback.start

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback.start
```

````

````{py:method} stop() -> None
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback.stop

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback.stop
```

````

````{py:method} update(overall_completed: int, policy_updates: typing.Dict[str, typing.Dict[str, typing.Any]], new_daily_data: typing.List[typing.Dict[str, typing.Any]]) -> None
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback.update

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback.update
```

````

````{py:method} _render_layout() -> rich.layout.Layout
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._render_layout

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._render_layout
```

````

````{py:method} _generate_chart() -> rich.panel.Panel
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._generate_chart

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._generate_chart
```

````

````{py:method} _generate_metrics_table() -> rich.panel.Panel
:canonical: src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._generate_metrics_table

```{autodoc2-docstring} src.pipeline.callbacks.simulation.simulation_display.SimulationDisplayCallback._generate_metrics_table
```

````

`````
