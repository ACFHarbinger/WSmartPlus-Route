# {py:mod}`src.configs.tracking`

```{py:module} src.configs.tracking
```

```{autodoc2-docstring} src.configs.tracking
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrackingConfig <src.configs.tracking.TrackingConfig>`
  - ```{autodoc2-docstring} src.configs.tracking.TrackingConfig
    :summary:
    ```
````

### API

`````{py:class} TrackingConfig
:canonical: src.configs.tracking.TrackingConfig

```{autodoc2-docstring} src.configs.tracking.TrackingConfig
```

````{py:attribute} wst_tracking_uri
:canonical: src.configs.tracking.TrackingConfig.wst_tracking_uri
:type: str
:value: >
   None

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.wst_tracking_uri
```

````

````{py:attribute} mlflow_enabled
:canonical: src.configs.tracking.TrackingConfig.mlflow_enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.mlflow_enabled
```

````

````{py:attribute} mlflow_tracking_uri
:canonical: src.configs.tracking.TrackingConfig.mlflow_tracking_uri
:type: str
:value: >
   None

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.mlflow_tracking_uri
```

````

````{py:attribute} mlflow_experiment_name
:canonical: src.configs.tracking.TrackingConfig.mlflow_experiment_name
:type: str
:value: >
   'wsmart-route'

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.mlflow_experiment_name
```

````

````{py:attribute} mlflow_run_name
:canonical: src.configs.tracking.TrackingConfig.mlflow_run_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.mlflow_run_name
```

````

````{py:attribute} ray_tune_storage_path
:canonical: src.configs.tracking.TrackingConfig.ray_tune_storage_path
:type: str
:value: >
   'ray_results'

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.ray_tune_storage_path
```

````

````{py:attribute} ray_tune_mlflow_enabled
:canonical: src.configs.tracking.TrackingConfig.ray_tune_mlflow_enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.ray_tune_mlflow_enabled
```

````

````{py:attribute} zenml_enabled
:canonical: src.configs.tracking.TrackingConfig.zenml_enabled
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.zenml_enabled
```

````

````{py:attribute} zenml_store_url
:canonical: src.configs.tracking.TrackingConfig.zenml_store_url
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.zenml_store_url
```

````

````{py:attribute} zenml_stack_name
:canonical: src.configs.tracking.TrackingConfig.zenml_stack_name
:type: str
:value: >
   'wsmart-route-stack'

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.zenml_stack_name
```

````

`````
