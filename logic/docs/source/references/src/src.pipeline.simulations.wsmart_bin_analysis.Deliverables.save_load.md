# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`save_container_structured <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_container_structured>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_container_structured
    :summary:
    ```
* - {py:obj}`load_container_structured <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_container_structured>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_container_structured
    :summary:
    ```
* - {py:obj}`save_id_containers <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_id_containers>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_id_containers
    :summary:
    ```
* - {py:obj}`load_id_containers <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_id_containers>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_id_containers
    :summary:
    ```
* - {py:obj}`save_rate_series <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_rate_series>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_rate_series
    :summary:
    ```
* - {py:obj}`load_info <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_info>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_info
    :summary:
    ```
* - {py:obj}`load_rate_series <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_series>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_series
    :summary:
    ```
* - {py:obj}`load_rate_global_wrapper <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_global_wrapper>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_global_wrapper
    :summary:
    ```
* - {py:obj}`verify_names <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.verify_names>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.verify_names
    :summary:
    ```
* - {py:obj}`container_names <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.container_names>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.container_names
    :summary:
    ```
````

### API

````{py:function} save_container_structured(id: int, container: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container, ver=None, path=None, names=None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_container_structured

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_container_structured
```
````

````{py:function} load_container_structured(id=None, ver=None, path=None, names=None) -> src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_container_structured

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_container_structured
```
````

````{py:function} save_id_containers(id_list: list[int], path=None, name=None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_id_containers

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_id_containers
```
````

````{py:function} load_id_containers(path=None, name='ids.csv') -> list[int]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_id_containers

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_id_containers
```
````

````{py:function} save_rate_series(id: int, container: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container, rate_type: str, freq: str, path=None, names=None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_rate_series

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.save_rate_series
```
````

````{py:function} load_info(id=None, ver=None, path=None, name=None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_info

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_info
```
````

````{py:function} load_rate_series(id: int, rate_type: str, path=None, name=None) -> dict[str, typing.Union[int, pandas.DataFrame]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_series

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_series
```
````

````{py:function} load_rate_global_wrapper(rate_list: list[dict]) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_global_wrapper

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.load_rate_global_wrapper
```
````

````{py:function} verify_names(id=None, ver=None, path=None, names=None) -> tuple[list[str], str]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.verify_names

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.verify_names
```
````

````{py:function} container_names(id: int, ver: str) -> list[str]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.container_names

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.save_load.container_names
```
````
