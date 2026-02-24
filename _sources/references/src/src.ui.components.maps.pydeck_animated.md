# {py:mod}`src.ui.components.maps.pydeck_animated`

```{py:module} src.ui.components.maps.pydeck_animated
```

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_pydeck_animated_map <src.ui.components.maps.pydeck_animated.render_pydeck_animated_map>`
  - ```{autodoc2-docstring} src.ui.components.maps.pydeck_animated.render_pydeck_animated_map
    :summary:
    ```
* - {py:obj}`_load_jsonl <src.ui.components.maps.pydeck_animated._load_jsonl>`
  - ```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._load_jsonl
    :summary:
    ```
* - {py:obj}`_reconstruct_fill_levels <src.ui.components.maps.pydeck_animated._reconstruct_fill_levels>`
  - ```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._reconstruct_fill_levels
    :summary:
    ```
* - {py:obj}`_build_column_data <src.ui.components.maps.pydeck_animated._build_column_data>`
  - ```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._build_column_data
    :summary:
    ```
* - {py:obj}`_build_arc_data <src.ui.components.maps.pydeck_animated._build_arc_data>`
  - ```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._build_arc_data
    :summary:
    ```
* - {py:obj}`_render_day_kpis <src.ui.components.maps.pydeck_animated._render_day_kpis>`
  - ```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._render_day_kpis
    :summary:
    ```
* - {py:obj}`_render_static_bins <src.ui.components.maps.pydeck_animated._render_static_bins>`
  - ```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._render_static_bins
    :summary:
    ```
````

### API

````{py:function} render_pydeck_animated_map(bin_locations: typing.List[typing.Dict[str, float]], simulation_log_path: str, policy_name: str = 'gurobi', map_center_lat: float = 51.0, map_center_lon: float = 4.0, initial_zoom: int = 12, map_style: str = 'road', column_radius: int = 50, column_elevation_scale: int = 10, arc_width: int = 3, daily_fill_increment: float = 5.0, height: int = 600, title: str = 'Simulation Animation') -> None
:canonical: src.ui.components.maps.pydeck_animated.render_pydeck_animated_map

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated.render_pydeck_animated_map
```
````

````{py:function} _load_jsonl(path: pathlib.Path, policy_filter: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.components.maps.pydeck_animated._load_jsonl

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._load_jsonl
```
````

````{py:function} _reconstruct_fill_levels(bins: typing.List[typing.Dict[str, float]], records: typing.List[typing.Dict[str, typing.Any]], up_to_day: int, daily_increment: float) -> typing.List[float]
:canonical: src.ui.components.maps.pydeck_animated._reconstruct_fill_levels

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._reconstruct_fill_levels
```
````

````{py:function} _build_column_data(bins: typing.List[typing.Dict[str, float]], fill_levels: typing.List[float]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.components.maps.pydeck_animated._build_column_data

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._build_column_data
```
````

````{py:function} _build_arc_data(bins: typing.List[typing.Dict[str, float]], records: typing.List[typing.Dict[str, typing.Any]], day: int) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.components.maps.pydeck_animated._build_arc_data

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._build_arc_data
```
````

````{py:function} _render_day_kpis(record: typing.Dict[str, typing.Any]) -> None
:canonical: src.ui.components.maps.pydeck_animated._render_day_kpis

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._render_day_kpis
```
````

````{py:function} _render_static_bins(bins: typing.List[typing.Dict[str, float]], lat: float, lon: float, zoom: int, pdk: typing.Any) -> None
:canonical: src.ui.components.maps.pydeck_animated._render_static_bins

```{autodoc2-docstring} src.ui.components.maps.pydeck_animated._render_static_bins
```
````
