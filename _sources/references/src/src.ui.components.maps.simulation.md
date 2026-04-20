# {py:mod}`src.ui.components.maps.simulation`

```{py:module} src.ui.components.maps.simulation
```

```{autodoc2-docstring} src.ui.components.maps.simulation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_add_bin_marker <src.ui.components.maps.simulation._add_bin_marker>`
  - ```{autodoc2-docstring} src.ui.components.maps.simulation._add_bin_marker
    :summary:
    ```
* - {py:obj}`create_simulation_map <src.ui.components.maps.simulation.create_simulation_map>`
  - ```{autodoc2-docstring} src.ui.components.maps.simulation.create_simulation_map
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`template_dir <src.ui.components.maps.simulation.template_dir>`
  - ```{autodoc2-docstring} src.ui.components.maps.simulation.template_dir
    :summary:
    ```
* - {py:obj}`jinja_env <src.ui.components.maps.simulation.jinja_env>`
  - ```{autodoc2-docstring} src.ui.components.maps.simulation.jinja_env
    :summary:
    ```
* - {py:obj}`DEPOT_POPUP_TEMPLATE <src.ui.components.maps.simulation.DEPOT_POPUP_TEMPLATE>`
  - ```{autodoc2-docstring} src.ui.components.maps.simulation.DEPOT_POPUP_TEMPLATE
    :summary:
    ```
* - {py:obj}`BIN_POPUP_TEMPLATE <src.ui.components.maps.simulation.BIN_POPUP_TEMPLATE>`
  - ```{autodoc2-docstring} src.ui.components.maps.simulation.BIN_POPUP_TEMPLATE
    :summary:
    ```
````

### API

````{py:data} template_dir
:canonical: src.ui.components.maps.simulation.template_dir
:value: >
   'join(...)'

```{autodoc2-docstring} src.ui.components.maps.simulation.template_dir
```

````

````{py:data} jinja_env
:canonical: src.ui.components.maps.simulation.jinja_env
:value: >
   'Environment(...)'

```{autodoc2-docstring} src.ui.components.maps.simulation.jinja_env
```

````

````{py:data} DEPOT_POPUP_TEMPLATE
:canonical: src.ui.components.maps.simulation.DEPOT_POPUP_TEMPLATE
:value: >
   'get_template(...)'

```{autodoc2-docstring} src.ui.components.maps.simulation.DEPOT_POPUP_TEMPLATE
```

````

````{py:data} BIN_POPUP_TEMPLATE
:canonical: src.ui.components.maps.simulation.BIN_POPUP_TEMPLATE
:value: >
   'get_template(...)'

```{autodoc2-docstring} src.ui.components.maps.simulation.BIN_POPUP_TEMPLATE
```

````

````{py:function} _add_bin_marker(m: folium.Map, point: typing.Dict[str, typing.Any], bin_id: int, bin_states: typing.Optional[typing.List[float]], collected_set: set, mandatory_set: set, toured_ids: set, dataset_id: typing.Optional[int] = None) -> None
:canonical: src.ui.components.maps.simulation._add_bin_marker

```{autodoc2-docstring} src.ui.components.maps.simulation._add_bin_marker
```
````

````{py:function} create_simulation_map(tour: typing.List[typing.Dict[str, typing.Any]], bin_states: typing.Optional[typing.List[float]] = None, served_indices: typing.Optional[typing.List[int]] = None, mandatory: typing.Optional[typing.List[int]] = None, all_bin_coords: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None, collected: typing.Optional[typing.List[float]] = None, vehicle_id: int = 0, show_route: bool = True, zoom_start: int = 12, distance_matrix: typing.Optional[pandas.DataFrame] = None, dist_strategy: str = 'hsd') -> folium.Map
:canonical: src.ui.components.maps.simulation.create_simulation_map

```{autodoc2-docstring} src.ui.components.maps.simulation.create_simulation_map
```
````
