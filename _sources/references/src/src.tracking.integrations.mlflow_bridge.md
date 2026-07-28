# {py:mod}`src.tracking.integrations.mlflow_bridge`

```{py:module} src.tracking.integrations.mlflow_bridge
```

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLflowBridge <src.tracking.integrations.mlflow_bridge.MLflowBridge>`
  - ```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge
    :summary:
    ```
````

### API

`````{py:class} MLflowBridge(mlflow_tracking_uri: str, experiment_name: str, run_name: typing.Optional[str] = None, tags: typing.Optional[typing.Dict[str, str]] = None)
:canonical: src.tracking.integrations.mlflow_bridge.MLflowBridge

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge.__init__
```

````{py:method} log_metric(key: str, value: float, step: int) -> None
:canonical: src.tracking.integrations.mlflow_bridge.MLflowBridge.log_metric

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge.log_metric
```

````

````{py:method} log_params(params: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.integrations.mlflow_bridge.MLflowBridge.log_params

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge.log_params
```

````

````{py:method} log_artifact(path: str) -> None
:canonical: src.tracking.integrations.mlflow_bridge.MLflowBridge.log_artifact

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge.log_artifact
```

````

````{py:method} finish(status: str = 'completed') -> None
:canonical: src.tracking.integrations.mlflow_bridge.MLflowBridge.finish

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge.finish
```

````

````{py:method} attach(run: logic.src.tracking.core.run.Run, mlflow_tracking_uri: str, experiment_name: str, run_name: typing.Optional[str] = None, tags: typing.Optional[typing.Dict[str, str]] = None) -> src.tracking.integrations.mlflow_bridge.MLflowBridge
:canonical: src.tracking.integrations.mlflow_bridge.MLflowBridge.attach
:classmethod:

```{autodoc2-docstring} src.tracking.integrations.mlflow_bridge.MLflowBridge.attach
```

````

`````
