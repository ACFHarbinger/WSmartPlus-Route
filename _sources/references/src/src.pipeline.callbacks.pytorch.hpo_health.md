# {py:mod}`src.pipeline.callbacks.pytorch.hpo_health`

```{py:module} src.pipeline.callbacks.pytorch.hpo_health
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HpoHealthMetricsCallback <src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_dehb_health_penalty <src.pipeline.callbacks.pytorch.hpo_health.apply_dehb_health_penalty>`
  - ```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.apply_dehb_health_penalty
    :summary:
    ```
````

### API

`````{py:class} HpoHealthMetricsCallback(trial: typing.Optional[typing.Any] = None, max_grad_norm: float = 100.0, min_entropy: float = 0.01, prune_on_unhealthy: bool = True, reward_key: str = 'val/reward', grad_norm_key: str = 'train/grad_norm', entropy_key: str = 'train/entropy', report_to_ray: bool = False)
:canonical: src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback

Bases: {py:obj}`pytorch_lightning.callbacks.Callback`

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback.__init__
```

````{py:method} on_validation_epoch_end(trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule) -> None
:canonical: src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback.on_validation_epoch_end

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback.on_validation_epoch_end
```

````

````{py:method} _log_to_tracker(grad_norm: typing.Optional[float], entropy: typing.Optional[float], reward: typing.Optional[float], epoch: int) -> None
:canonical: src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._log_to_tracker

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._log_to_tracker
```

````

````{py:method} _report_to_ray(grad_norm: typing.Optional[float], entropy: typing.Optional[float], reward: typing.Optional[float], epoch: int) -> None
:canonical: src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._report_to_ray

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._report_to_ray
```

````

````{py:method} _report_to_optuna(trainer: pytorch_lightning.Trainer, grad_norm: typing.Optional[float], entropy: typing.Optional[float], reward: typing.Optional[float], epoch: int) -> None
:canonical: src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._report_to_optuna

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._report_to_optuna
```

````

````{py:method} _metric_value(metrics: typing.Dict[str, typing.Any], key: str) -> typing.Optional[float]
:canonical: src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._metric_value
:staticmethod:

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.HpoHealthMetricsCallback._metric_value
```

````

`````

````{py:function} apply_dehb_health_penalty(fitness: float, grad_norm: typing.Optional[float], entropy: typing.Optional[float], max_grad_norm: float = 100.0, min_entropy: float = 0.01) -> float
:canonical: src.pipeline.callbacks.pytorch.hpo_health.apply_dehb_health_penalty

```{autodoc2-docstring} src.pipeline.callbacks.pytorch.hpo_health.apply_dehb_health_penalty
```
````
