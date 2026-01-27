# {py:mod}`src.pipeline.dashboard`

```{py:module} src.pipeline.dashboard
```

```{autodoc2-docstring} src.pipeline.dashboard
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`haversine <src.pipeline.dashboard.haversine>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.haversine
    :summary:
    ```
* - {py:obj}`get_lightning_runs <src.pipeline.dashboard.get_lightning_runs>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.get_lightning_runs
    :summary:
    ```
* - {py:obj}`load_training_data <src.pipeline.dashboard.load_training_data>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.load_training_data
    :summary:
    ```
* - {py:obj}`get_simulation_logs <src.pipeline.dashboard.get_simulation_logs>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.get_simulation_logs
    :summary:
    ```
* - {py:obj}`parse_simulation_log_line <src.pipeline.dashboard.parse_simulation_log_line>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.parse_simulation_log_line
    :summary:
    ```
* - {py:obj}`load_simulation_history <src.pipeline.dashboard.load_simulation_history>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.load_simulation_history
    :summary:
    ```
* - {py:obj}`load_distance_matrix <src.pipeline.dashboard.load_distance_matrix>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.load_distance_matrix
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`mode <src.pipeline.dashboard.mode>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.mode
    :summary:
    ```
* - {py:obj}`realtime_mode <src.pipeline.dashboard.realtime_mode>`
  - ```{autodoc2-docstring} src.pipeline.dashboard.realtime_mode
    :summary:
    ```
````

### API

````{py:function} haversine(lat1, lon1, lat2, lon2)
:canonical: src.pipeline.dashboard.haversine

```{autodoc2-docstring} src.pipeline.dashboard.haversine
```
````

````{py:function} get_lightning_runs(root_dir: str = 'lightning_logs') -> typing.List[str]
:canonical: src.pipeline.dashboard.get_lightning_runs

```{autodoc2-docstring} src.pipeline.dashboard.get_lightning_runs
```
````

````{py:function} load_training_data(run_dir: str)
:canonical: src.pipeline.dashboard.load_training_data

```{autodoc2-docstring} src.pipeline.dashboard.load_training_data
```
````

````{py:function} get_simulation_logs(root_dir: str = 'assets/output') -> typing.List[str]
:canonical: src.pipeline.dashboard.get_simulation_logs

```{autodoc2-docstring} src.pipeline.dashboard.get_simulation_logs
```
````

````{py:function} parse_simulation_log_line(line: str) -> typing.Optional[typing.Dict[str, typing.Any]]
:canonical: src.pipeline.dashboard.parse_simulation_log_line

```{autodoc2-docstring} src.pipeline.dashboard.parse_simulation_log_line
```
````

````{py:function} load_simulation_history(log_path: str)
:canonical: src.pipeline.dashboard.load_simulation_history

```{autodoc2-docstring} src.pipeline.dashboard.load_simulation_history
```
````

````{py:function} load_distance_matrix(area: str, size: int, waste_type: str = 'plastic')
:canonical: src.pipeline.dashboard.load_distance_matrix

```{autodoc2-docstring} src.pipeline.dashboard.load_distance_matrix
```
````

````{py:data} mode
:canonical: src.pipeline.dashboard.mode
:value: >
   'radio(...)'

```{autodoc2-docstring} src.pipeline.dashboard.mode
```

````

````{py:data} realtime_mode
:canonical: src.pipeline.dashboard.realtime_mode
:value: >
   'toggle(...)'

```{autodoc2-docstring} src.pipeline.dashboard.realtime_mode
```

````
