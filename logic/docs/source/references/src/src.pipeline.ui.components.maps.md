# {py:mod}`src.pipeline.ui.components.maps`

```{py:module} src.pipeline.ui.components.maps
```

```{autodoc2-docstring} src.pipeline.ui.components.maps
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_map_center <src.pipeline.ui.components.maps.get_map_center>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.get_map_center
    :summary:
    ```
* - {py:obj}`load_distance_matrix <src.pipeline.ui.components.maps.load_distance_matrix>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.load_distance_matrix
    :summary:
    ```
* - {py:obj}`create_simulation_map <src.pipeline.ui.components.maps.create_simulation_map>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.create_simulation_map
    :summary:
    ```
* - {py:obj}`create_multi_route_map <src.pipeline.ui.components.maps.create_multi_route_map>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.create_multi_route_map
    :summary:
    ```
* - {py:obj}`create_bin_heatmap <src.pipeline.ui.components.maps.create_bin_heatmap>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.create_bin_heatmap
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ROUTE_COLORS <src.pipeline.ui.components.maps.ROUTE_COLORS>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.ROUTE_COLORS
    :summary:
    ```
* - {py:obj}`BIN_COLORS <src.pipeline.ui.components.maps.BIN_COLORS>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.maps.BIN_COLORS
    :summary:
    ```
````

### API

````{py:data} ROUTE_COLORS
:canonical: src.pipeline.ui.components.maps.ROUTE_COLORS
:value: >
   ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

```{autodoc2-docstring} src.pipeline.ui.components.maps.ROUTE_COLORS
```

````

````{py:data} BIN_COLORS
:canonical: src.pipeline.ui.components.maps.BIN_COLORS
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.components.maps.BIN_COLORS
```

````

````{py:function} get_map_center(tour: typing.List[typing.Dict[str, typing.Any]]) -> typing.Tuple[float, float]
:canonical: src.pipeline.ui.components.maps.get_map_center

```{autodoc2-docstring} src.pipeline.ui.components.maps.get_map_center
```
````

````{py:function} load_distance_matrix(instance_name: str = 'riomaior') -> typing.Optional[pandas.DataFrame]
:canonical: src.pipeline.ui.components.maps.load_distance_matrix

```{autodoc2-docstring} src.pipeline.ui.components.maps.load_distance_matrix
```
````

````{py:function} create_simulation_map(tour: typing.List[typing.Dict[str, typing.Any]], bin_states: typing.Optional[typing.List[float]] = None, served_indices: typing.Optional[typing.List[int]] = None, vehicle_id: int = 0, show_route: bool = True, zoom_start: int = 12, distance_matrix: typing.Optional[pandas.DataFrame] = None, dist_strategy: str = 'hsd') -> folium.Map
:canonical: src.pipeline.ui.components.maps.create_simulation_map

```{autodoc2-docstring} src.pipeline.ui.components.maps.create_simulation_map
```
````

````{py:function} create_multi_route_map(routes: typing.List[typing.List[typing.Dict[str, typing.Any]]], bin_states: typing.Optional[typing.List[float]] = None, zoom_start: int = 12) -> folium.Map
:canonical: src.pipeline.ui.components.maps.create_multi_route_map

```{autodoc2-docstring} src.pipeline.ui.components.maps.create_multi_route_map
```
````

````{py:function} create_bin_heatmap(bin_locations: typing.List[typing.Dict[str, typing.Any]], bin_states: typing.List[float]) -> folium.Map
:canonical: src.pipeline.ui.components.maps.create_bin_heatmap

```{autodoc2-docstring} src.pipeline.ui.components.maps.create_bin_heatmap
```
````
