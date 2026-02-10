# {py:mod}`src.pipeline.callbacks.simulation_display`

```{py:module} src.pipeline.callbacks.simulation_display
```

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimulationDisplay <src.pipeline.callbacks.simulation_display.SimulationDisplay>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay
    :summary:
    ```
````

### API

`````{py:class} SimulationDisplay(policies: typing.List[str], n_samples: int, total_days: int, chart_metric: str = 'profit', refresh_rate: int = 2, theme: str = 'dark')
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay.__init__
```

````{py:method} _init_layout() -> None
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay._init_layout

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay._init_layout
```

````

````{py:method} start() -> None
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay.start

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay.start
```

````

````{py:method} stop() -> None
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay.stop

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay.stop
```

````

````{py:method} update(overall_completed: int, policy_updates: typing.Dict[str, typing.Dict[str, typing.Any]], new_daily_data: typing.List[typing.Dict[str, typing.Any]]) -> None
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay.update

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay.update
```

````

````{py:method} _render_layout() -> rich.layout.Layout
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay._render_layout

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay._render_layout
```

````

````{py:method} _generate_chart() -> rich.panel.Panel
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay._generate_chart

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay._generate_chart
```

````

````{py:method} _generate_metrics_table() -> rich.panel.Panel
:canonical: src.pipeline.callbacks.simulation_display.SimulationDisplay._generate_metrics_table

```{autodoc2-docstring} src.pipeline.callbacks.simulation_display.SimulationDisplay._generate_metrics_table
```

````

`````
