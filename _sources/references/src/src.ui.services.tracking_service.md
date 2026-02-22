# {py:mod}`src.ui.services.tracking_service`

```{py:module} src.ui.services.tracking_service
```

```{autodoc2-docstring} src.ui.services.tracking_service
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_tracking_uri <src.ui.services.tracking_service._get_tracking_uri>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service._get_tracking_uri
    :summary:
    ```
* - {py:obj}`_open_store <src.ui.services.tracking_service._open_store>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service._open_store
    :summary:
    ```
* - {py:obj}`load_tracking_runs <src.ui.services.tracking_service.load_tracking_runs>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_tracking_runs
    :summary:
    ```
* - {py:obj}`load_run_metrics <src.ui.services.tracking_service.load_run_metrics>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_run_metrics
    :summary:
    ```
* - {py:obj}`load_run_params <src.ui.services.tracking_service.load_run_params>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_run_params
    :summary:
    ```
* - {py:obj}`list_metric_keys <src.ui.services.tracking_service.list_metric_keys>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.list_metric_keys
    :summary:
    ```
* - {py:obj}`load_run_tags <src.ui.services.tracking_service.load_run_tags>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_run_tags
    :summary:
    ```
* - {py:obj}`load_run_artifacts <src.ui.services.tracking_service.load_run_artifacts>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_run_artifacts
    :summary:
    ```
* - {py:obj}`load_mlflow_runs <src.ui.services.tracking_service.load_mlflow_runs>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_mlflow_runs
    :summary:
    ```
* - {py:obj}`load_mlflow_metric_history <src.ui.services.tracking_service.load_mlflow_metric_history>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_mlflow_metric_history
    :summary:
    ```
* - {py:obj}`list_mlflow_metric_keys <src.ui.services.tracking_service.list_mlflow_metric_keys>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.list_mlflow_metric_keys
    :summary:
    ```
* - {py:obj}`load_zenml_pipeline_runs <src.ui.services.tracking_service.load_zenml_pipeline_runs>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_zenml_pipeline_runs
    :summary:
    ```
* - {py:obj}`load_zenml_run_steps <src.ui.services.tracking_service.load_zenml_run_steps>`
  - ```{autodoc2-docstring} src.ui.services.tracking_service.load_zenml_run_steps
    :summary:
    ```
````

### API

````{py:function} _get_tracking_uri(tracking_uri: typing.Optional[str]) -> str
:canonical: src.ui.services.tracking_service._get_tracking_uri

```{autodoc2-docstring} src.ui.services.tracking_service._get_tracking_uri
```
````

````{py:function} _open_store(tracking_uri: typing.Optional[str]) -> typing.Optional[typing.Any]
:canonical: src.ui.services.tracking_service._open_store

```{autodoc2-docstring} src.ui.services.tracking_service._open_store
```
````

````{py:function} load_tracking_runs(tracking_uri: typing.Optional[str] = None, run_type: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.services.tracking_service.load_tracking_runs

```{autodoc2-docstring} src.ui.services.tracking_service.load_tracking_runs
```
````

````{py:function} load_run_metrics(run_id: str, metric_key: str, tracking_uri: typing.Optional[str] = None) -> pandas.DataFrame
:canonical: src.ui.services.tracking_service.load_run_metrics

```{autodoc2-docstring} src.ui.services.tracking_service.load_run_metrics
```
````

````{py:function} load_run_params(run_id: str, tracking_uri: typing.Optional[str] = None) -> typing.Dict[str, typing.Any]
:canonical: src.ui.services.tracking_service.load_run_params

```{autodoc2-docstring} src.ui.services.tracking_service.load_run_params
```
````

````{py:function} list_metric_keys(run_id: str, tracking_uri: typing.Optional[str] = None) -> typing.List[str]
:canonical: src.ui.services.tracking_service.list_metric_keys

```{autodoc2-docstring} src.ui.services.tracking_service.list_metric_keys
```
````

````{py:function} load_run_tags(run_id: str, tracking_uri: typing.Optional[str] = None) -> typing.Dict[str, str]
:canonical: src.ui.services.tracking_service.load_run_tags

```{autodoc2-docstring} src.ui.services.tracking_service.load_run_tags
```
````

````{py:function} load_run_artifacts(run_id: str, tracking_uri: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.services.tracking_service.load_run_artifacts

```{autodoc2-docstring} src.ui.services.tracking_service.load_run_artifacts
```
````

````{py:function} load_mlflow_runs(tracking_uri: str = 'mlruns', experiment_name: typing.Optional[str] = None) -> typing.Union[pandas.DataFrame, typing.List[mlflow.entities.Run]]
:canonical: src.ui.services.tracking_service.load_mlflow_runs

```{autodoc2-docstring} src.ui.services.tracking_service.load_mlflow_runs
```
````

````{py:function} load_mlflow_metric_history(run_id: str, metric_key: str, tracking_uri: str = 'mlruns') -> pandas.DataFrame
:canonical: src.ui.services.tracking_service.load_mlflow_metric_history

```{autodoc2-docstring} src.ui.services.tracking_service.load_mlflow_metric_history
```
````

````{py:function} list_mlflow_metric_keys(run_id: str, tracking_uri: str = 'mlruns') -> typing.List[str]
:canonical: src.ui.services.tracking_service.list_mlflow_metric_keys

```{autodoc2-docstring} src.ui.services.tracking_service.list_mlflow_metric_keys
```
````

````{py:function} load_zenml_pipeline_runs() -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.services.tracking_service.load_zenml_pipeline_runs

```{autodoc2-docstring} src.ui.services.tracking_service.load_zenml_pipeline_runs
```
````

````{py:function} load_zenml_run_steps(run_id: str) -> typing.List[typing.Dict[str, typing.Any]]
:canonical: src.ui.services.tracking_service.load_zenml_run_steps

```{autodoc2-docstring} src.ui.services.tracking_service.load_zenml_run_steps
```
````
