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

````{py:attribute} wandb_mode
:canonical: src.configs.tracking.TrackingConfig.wandb_mode
:type: str
:value: >
   'offline'

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.wandb_mode
```

````

````{py:attribute} no_tensorboard
:canonical: src.configs.tracking.TrackingConfig.no_tensorboard
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.no_tensorboard
```

````

````{py:attribute} no_progress_bar
:canonical: src.configs.tracking.TrackingConfig.no_progress_bar
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.no_progress_bar
```

````

````{py:attribute} log_dir
:canonical: src.configs.tracking.TrackingConfig.log_dir
:type: str
:value: >
   'logs'

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_dir
```

````

````{py:attribute} verbose
:canonical: src.configs.tracking.TrackingConfig.verbose
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.verbose
```

````

````{py:attribute} profile
:canonical: src.configs.tracking.TrackingConfig.profile
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.profile
```

````

````{py:attribute} log_step
:canonical: src.configs.tracking.TrackingConfig.log_step
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_step
```

````

````{py:attribute} log_level
:canonical: src.configs.tracking.TrackingConfig.log_level
:type: str
:value: >
   'INFO'

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_level
```

````

````{py:attribute} real_time_log
:canonical: src.configs.tracking.TrackingConfig.real_time_log
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.real_time_log
```

````

````{py:attribute} log_file
:canonical: src.configs.tracking.TrackingConfig.log_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_file
```

````

````{py:attribute} log_gradients
:canonical: src.configs.tracking.TrackingConfig.log_gradients
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_gradients
```

````

````{py:attribute} log_weights
:canonical: src.configs.tracking.TrackingConfig.log_weights
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_weights
```

````

````{py:attribute} log_activations
:canonical: src.configs.tracking.TrackingConfig.log_activations
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_activations
```

````

````{py:attribute} log_attention
:canonical: src.configs.tracking.TrackingConfig.log_attention
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_attention
```

````

````{py:attribute} log_memory
:canonical: src.configs.tracking.TrackingConfig.log_memory
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_memory
```

````

````{py:attribute} log_throughput
:canonical: src.configs.tracking.TrackingConfig.log_throughput
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_throughput
```

````

````{py:attribute} log_embeddings
:canonical: src.configs.tracking.TrackingConfig.log_embeddings
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_embeddings
```

````

````{py:attribute} log_loss_landscape
:canonical: src.configs.tracking.TrackingConfig.log_loss_landscape
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_loss_landscape
```

````

````{py:attribute} log_attention_heatmaps
:canonical: src.configs.tracking.TrackingConfig.log_attention_heatmaps
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_attention_heatmaps
```

````

````{py:attribute} log_profiling_report
:canonical: src.configs.tracking.TrackingConfig.log_profiling_report
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.log_profiling_report
```

````

````{py:attribute} nan_guard
:canonical: src.configs.tracking.TrackingConfig.nan_guard
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.nan_guard
```

````

````{py:attribute} viz_every_n_epochs
:canonical: src.configs.tracking.TrackingConfig.viz_every_n_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.viz_every_n_epochs
```

````

````{py:attribute} profiler_buffer_size
:canonical: src.configs.tracking.TrackingConfig.profiler_buffer_size
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.tracking.TrackingConfig.profiler_buffer_size
```

````

`````
