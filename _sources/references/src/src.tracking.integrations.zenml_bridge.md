# {py:mod}`src.tracking.integrations.zenml_bridge`

```{py:module} src.tracking.integrations.zenml_bridge
```

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ZenMLBridge <src.tracking.integrations.zenml_bridge.ZenMLBridge>`
  - ```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`configure_zenml_stack <src.tracking.integrations.zenml_bridge.configure_zenml_stack>`
  - ```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.configure_zenml_stack
    :summary:
    ```
* - {py:obj}`extract_zenml_step_output <src.tracking.integrations.zenml_bridge.extract_zenml_step_output>`
  - ```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.extract_zenml_step_output
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.tracking.integrations.zenml_bridge.logger>`
  - ```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.tracking.integrations.zenml_bridge.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.logger
```

````

`````{py:class} ZenMLBridge()
:canonical: src.tracking.integrations.zenml_bridge.ZenMLBridge

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge.__init__
```

````{py:method} _has_active_run() -> bool
:canonical: src.tracking.integrations.zenml_bridge.ZenMLBridge._has_active_run

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge._has_active_run
```

````

````{py:method} log_metric(key: str, value: float, step: int) -> None
:canonical: src.tracking.integrations.zenml_bridge.ZenMLBridge.log_metric

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge.log_metric
```

````

````{py:method} log_params(params: typing.Dict[str, typing.Any]) -> None
:canonical: src.tracking.integrations.zenml_bridge.ZenMLBridge.log_params

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge.log_params
```

````

````{py:method} log_artifact(path: str) -> None
:canonical: src.tracking.integrations.zenml_bridge.ZenMLBridge.log_artifact

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge.log_artifact
```

````

````{py:method} finish(status: str = 'completed') -> None
:canonical: src.tracking.integrations.zenml_bridge.ZenMLBridge.finish

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge.finish
```

````

````{py:method} attach(run: logic.src.tracking.core.run.Run) -> src.tracking.integrations.zenml_bridge.ZenMLBridge
:canonical: src.tracking.integrations.zenml_bridge.ZenMLBridge.attach
:classmethod:

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.ZenMLBridge.attach
```

````

`````

````{py:function} configure_zenml_stack(mlflow_tracking_uri: str, stack_name: str = 'wsmart-route-stack') -> bool
:canonical: src.tracking.integrations.zenml_bridge.configure_zenml_stack

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.configure_zenml_stack
```
````

````{py:function} extract_zenml_step_output(pipeline_name: str, step_name: str) -> typing.Optional[typing.Any]
:canonical: src.tracking.integrations.zenml_bridge.extract_zenml_step_output

```{autodoc2-docstring} src.tracking.integrations.zenml_bridge.extract_zenml_step_output
```
````
