# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fix_collections_sensor <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.fix_collections_sensor>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.fix_collections_sensor
    :summary:
    ```
* - {py:obj}`get_overall_sensors_statistics <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.get_overall_sensors_statistics>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.get_overall_sensors_statistics
    :summary:
    ```
* - {py:obj}`filter_containers <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.filter_containers>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.filter_containers
    :summary:
    ```
* - {py:obj}`pre_process_container_metrics <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.pre_process_container_metrics>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.pre_process_container_metrics
    :summary:
    ```
* - {py:obj}`view_metrics <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.view_metrics>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.view_metrics
    :summary:
    ```
````

### API

````{py:function} fix_collections_sensor(container: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container, box_window: int, mv_thresh: int, min_days: int, dist_thresh: int, c_trash: int, max_fill: int, var_thresh: int, use: str, spear_thresh: int = None) -> tuple[float, list[src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG], src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.fix_collections_sensor

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.fix_collections_sensor
```
````

````{py:function} get_overall_sensors_statistics(containers_dict: dict) -> tuple[dict, dict]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.get_overall_sensors_statistics

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.get_overall_sensors_statistics
```
````

````{py:function} filter_containers(containers_dict: dict) -> dict
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.filter_containers

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.filter_containers
```
````

````{py:function} pre_process_container_metrics(containers_dict: dict, calc_spearman: bool = True)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.pre_process_container_metrics

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.pre_process_container_metrics
```
````

````{py:function} view_metrics(containers_dict: dict, box_window: int, mv_thresh: int, min_days: int, use: str)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.view_metrics

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.transform.view_metrics
```
````
