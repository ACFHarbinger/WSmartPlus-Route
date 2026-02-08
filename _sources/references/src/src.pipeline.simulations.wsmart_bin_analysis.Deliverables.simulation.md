# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Simulation <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation
    :summary:
    ```
````

### API

`````{py:class} Simulation(sim_type: str, ids: list, data_dir: str, train_split=None, start_date=None, end_date=None, rate_type=None, predictQ: bool = False, info_ver=None, names=None, savefit_name=None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation

Bases: {py:obj}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase`

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.__init__
```

````{py:method} pre_simulate_rates() -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.pre_simulate_rates

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.pre_simulate_rates
```

````

````{py:method} reset_simulation()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.reset_simulation

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.reset_simulation
```

````

````{py:method} get_current_step() -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray], typing.Optional[numpy.ndarray]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.get_current_step

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.get_current_step
```

````

````{py:method} make_collections(bins_index_list: list[int] = None) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.make_collections

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.make_collections
```

````

````{py:method} advance_timestep(date=None) -> tuple[int, typing.Optional[numpy.ndarray], typing.Optional[numpy.ndarray]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.advance_timestep

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.advance_timestep
```

````

`````
