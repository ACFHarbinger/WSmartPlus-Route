# {py:mod}`src.pipeline.simulations.hpo.hpo_handler`

```{py:module} src.pipeline.simulations.hpo.hpo_handler
```

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_metric_direction <src.pipeline.simulations.hpo.hpo_handler._metric_direction>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._metric_direction
    :summary:
    ```
* - {py:obj}`_extract_metric <src.pipeline.simulations.hpo.hpo_handler._extract_metric>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._extract_metric
    :summary:
    ```
* - {py:obj}`objective <src.pipeline.simulations.hpo.hpo_handler.objective>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.objective
    :summary:
    ```
* - {py:obj}`worker <src.pipeline.simulations.hpo.hpo_handler.worker>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.worker
    :summary:
    ```
* - {py:obj}`run_hpo_sim <src.pipeline.simulations.hpo.hpo_handler.run_hpo_sim>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.run_hpo_sim
    :summary:
    ```
* - {py:obj}`_select_pareto_representative <src.pipeline.simulations.hpo.hpo_handler._select_pareto_representative>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._select_pareto_representative
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.simulations.hpo.hpo_handler.logger>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.logger
    :summary:
    ```
* - {py:obj}`MIN_RECOMMENDED_SAMPLES <src.pipeline.simulations.hpo.hpo_handler.MIN_RECOMMENDED_SAMPLES>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.MIN_RECOMMENDED_SAMPLES
    :summary:
    ```
* - {py:obj}`_MINIMISE_METRICS <src.pipeline.simulations.hpo.hpo_handler._MINIMISE_METRICS>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._MINIMISE_METRICS
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.simulations.hpo.hpo_handler.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.logger
```

````

````{py:data} MIN_RECOMMENDED_SAMPLES
:canonical: src.pipeline.simulations.hpo.hpo_handler.MIN_RECOMMENDED_SAMPLES
:value: >
   5

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.MIN_RECOMMENDED_SAMPLES
```

````

````{py:data} _MINIMISE_METRICS
:canonical: src.pipeline.simulations.hpo.hpo_handler._MINIMISE_METRICS
:value: >
   None

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._MINIMISE_METRICS
```

````

````{py:function} _metric_direction(metric_name: str) -> str
:canonical: src.pipeline.simulations.hpo.hpo_handler._metric_direction

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._metric_direction
```
````

````{py:function} _extract_metric(log: typing.Dict, metric_name: str) -> float
:canonical: src.pipeline.simulations.hpo.hpo_handler._extract_metric

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._extract_metric
```
````

````{py:function} objective(trial: optuna.Trial, base_cfg: logic.src.configs.Config, data_size: int, lock: typing.Any) -> typing.Union[float, typing.Tuple[float, ...]]
:canonical: src.pipeline.simulations.hpo.hpo_handler.objective

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.objective
```
````

````{py:function} worker(study_name: str, storage_url: str, base_cfg_yaml: str, data_size: int, n_trials: int, lock: typing.Any) -> None
:canonical: src.pipeline.simulations.hpo.hpo_handler.worker

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.worker
```
````

````{py:function} run_hpo_sim(cfg: logic.src.configs.Config) -> typing.Union[float, typing.List[float]]
:canonical: src.pipeline.simulations.hpo.hpo_handler.run_hpo_sim

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler.run_hpo_sim
```
````

````{py:function} _select_pareto_representative(pareto_trials: typing.List[optuna.trial.FrozenTrial], directions: typing.List[str]) -> typing.Optional[optuna.trial.FrozenTrial]
:canonical: src.pipeline.simulations.hpo.hpo_handler._select_pareto_representative

```{autodoc2-docstring} src.pipeline.simulations.hpo.hpo_handler._select_pareto_representative
```
````
