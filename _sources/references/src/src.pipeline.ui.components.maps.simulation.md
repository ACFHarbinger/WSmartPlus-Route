# {py:mod}`src.pipeline.ui.components.maps.simulation`

```{py:module} src.pipeline.ui.components.maps.simulation
```

```{autodoc2-docstring} src.pipeline.ui.components.maps.simulation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_simulation_map <src.pipeline.ui.components.maps.simulation.create_simulation_map>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.simulation.create_simulation_map
    :summary:
    ```
````

### API

````{py:function} create_simulation_map(tour: typing.List[typing.Dict[str, typing.Any]], bin_states: typing.Optional[typing.List[float]] = None, served_indices: typing.Optional[typing.List[int]] = None, vehicle_id: int = 0, show_route: bool = True, zoom_start: int = 12, distance_matrix: typing.Optional[pandas.DataFrame] = None, dist_strategy: str = 'hsd') -> folium.Map
:canonical: src.pipeline.ui.components.maps.simulation.create_simulation_map

```{autodoc2-docstring} src.pipeline.ui.components.maps.simulation.create_simulation_map
```
````
