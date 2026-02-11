# {py:mod}`src.pipeline.ui.pages.simulation`

```{py:module} src.pipeline.ui.pages.simulation
```

```{autodoc2-docstring} src.pipeline.ui.pages.simulation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_normalize_tour_points <src.pipeline.ui.pages.simulation._normalize_tour_points>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._normalize_tour_points
    :summary:
    ```
* - {py:obj}`_filter_simulation_data <src.pipeline.ui.pages.simulation._filter_simulation_data>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._filter_simulation_data
    :summary:
    ```
* - {py:obj}`_render_kpi_dashboard <src.pipeline.ui.pages.simulation._render_kpi_dashboard>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_kpi_dashboard
    :summary:
    ```
* - {py:obj}`_render_policy_info <src.pipeline.ui.pages.simulation._render_policy_info>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_policy_info
    :summary:
    ```
* - {py:obj}`_load_custom_matrix <src.pipeline.ui.pages.simulation._load_custom_matrix>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._load_custom_matrix
    :summary:
    ```
* - {py:obj}`_render_map_view <src.pipeline.ui.pages.simulation._render_map_view>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_map_view
    :summary:
    ```
* - {py:obj}`_render_bin_heatmap <src.pipeline.ui.pages.simulation._render_bin_heatmap>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_bin_heatmap
    :summary:
    ```
* - {py:obj}`_render_bin_state_inspector <src.pipeline.ui.pages.simulation._render_bin_state_inspector>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_bin_state_inspector
    :summary:
    ```
* - {py:obj}`_render_collection_details <src.pipeline.ui.pages.simulation._render_collection_details>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_collection_details
    :summary:
    ```
* - {py:obj}`_render_tour_details <src.pipeline.ui.pages.simulation._render_tour_details>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_tour_details
    :summary:
    ```
* - {py:obj}`_render_metric_charts <src.pipeline.ui.pages.simulation._render_metric_charts>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_metric_charts
    :summary:
    ```
* - {py:obj}`_render_raw_data_view <src.pipeline.ui.pages.simulation._render_raw_data_view>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_raw_data_view
    :summary:
    ```
* - {py:obj}`render_simulation_visualizer <src.pipeline.ui.pages.simulation.render_simulation_visualizer>`
  - ```{autodoc2-docstring} src.pipeline.ui.pages.simulation.render_simulation_visualizer
    :summary:
    ```
````

### API

````{py:function} _normalize_tour_points(tour: typing.List[typing.Dict[str, typing.Any]]) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.ui.pages.simulation._normalize_tour_points

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._normalize_tour_points
```
````

````{py:function} _filter_simulation_data(entries: typing.List[typing.Any], controls: typing.Dict[str, typing.Any], day_range: typing.Tuple[int, int]) -> typing.List[typing.Any]
:canonical: src.pipeline.ui.pages.simulation._filter_simulation_data

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._filter_simulation_data
```
````

````{py:function} _render_kpi_dashboard(display_entry: typing.Any) -> None
:canonical: src.pipeline.ui.pages.simulation._render_kpi_dashboard

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_kpi_dashboard
```
````

````{py:function} _render_policy_info(display_entry: typing.Any) -> None
:canonical: src.pipeline.ui.pages.simulation._render_policy_info

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_policy_info
```
````

````{py:function} _load_custom_matrix(controls: typing.Dict[str, typing.Any]) -> typing.Any
:canonical: src.pipeline.ui.pages.simulation._load_custom_matrix

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._load_custom_matrix
```
````

````{py:function} _render_map_view(display_entry: typing.Any, controls: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.ui.pages.simulation._render_map_view

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_map_view
```
````

````{py:function} _render_bin_heatmap(display_entry: typing.Any) -> None
:canonical: src.pipeline.ui.pages.simulation._render_bin_heatmap

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_bin_heatmap
```
````

````{py:function} _render_bin_state_inspector(display_entry: typing.Any) -> None
:canonical: src.pipeline.ui.pages.simulation._render_bin_state_inspector

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_bin_state_inspector
```
````

````{py:function} _render_collection_details(display_entry: typing.Any) -> None
:canonical: src.pipeline.ui.pages.simulation._render_collection_details

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_collection_details
```
````

````{py:function} _render_tour_details(display_entry: typing.Any) -> None
:canonical: src.pipeline.ui.pages.simulation._render_tour_details

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_tour_details
```
````

````{py:function} _render_metric_charts(entries: typing.List[typing.Any], controls: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.ui.pages.simulation._render_metric_charts

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_metric_charts
```
````

````{py:function} _render_raw_data_view(display_entry: typing.Any) -> None
:canonical: src.pipeline.ui.pages.simulation._render_raw_data_view

```{autodoc2-docstring} src.pipeline.ui.pages.simulation._render_raw_data_view
```
````

````{py:function} render_simulation_visualizer() -> None
:canonical: src.pipeline.ui.pages.simulation.render_simulation_visualizer

```{autodoc2-docstring} src.pipeline.ui.pages.simulation.render_simulation_visualizer
```
````
