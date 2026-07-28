# {py:mod}`src.pipeline.features.train.hpo`

```{py:module} src.pipeline.features.train.hpo
```

```{autodoc2-docstring} src.pipeline.features.train.hpo
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`objective <src.pipeline.features.train.hpo.objective>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo.objective
    :summary:
    ```
* - {py:obj}`_ray_tune_objective <src.pipeline.features.train.hpo._ray_tune_objective>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo._ray_tune_objective
    :summary:
    ```
* - {py:obj}`run_hpo <src.pipeline.features.train.hpo.run_hpo>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo.run_hpo
    :summary:
    ```
* - {py:obj}`_mlflow_hpo_run <src.pipeline.features.train.hpo._mlflow_hpo_run>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo._mlflow_hpo_run
    :summary:
    ```
* - {py:obj}`_run_hpo_via_zenml <src.pipeline.features.train.hpo._run_hpo_via_zenml>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo._run_hpo_via_zenml
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.features.train.hpo.logger>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo.logger
    :summary:
    ```
* - {py:obj}`_RAY_TUNE_METHODS <src.pipeline.features.train.hpo._RAY_TUNE_METHODS>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo._RAY_TUNE_METHODS
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.features.train.hpo.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.train.hpo.logger
```

````

````{py:data} _RAY_TUNE_METHODS
:canonical: src.pipeline.features.train.hpo._RAY_TUNE_METHODS
:value: >
   None

```{autodoc2-docstring} src.pipeline.features.train.hpo._RAY_TUNE_METHODS
```

````

````{py:function} objective(trial: optuna.Trial, base_cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.hpo.objective

```{autodoc2-docstring} src.pipeline.features.train.hpo.objective
```
````

````{py:function} _ray_tune_objective(trial_cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.hpo._ray_tune_objective

```{autodoc2-docstring} src.pipeline.features.train.hpo._ray_tune_objective
```
````

````{py:function} run_hpo(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.hpo.run_hpo

```{autodoc2-docstring} src.pipeline.features.train.hpo.run_hpo
```
````

````{py:function} _mlflow_hpo_run(enabled: bool, tracking_uri: str, experiment_name: str, cfg: logic.src.configs.Config)
:canonical: src.pipeline.features.train.hpo._mlflow_hpo_run

```{autodoc2-docstring} src.pipeline.features.train.hpo._mlflow_hpo_run
```
````

````{py:function} _run_hpo_via_zenml(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.hpo._run_hpo_via_zenml

```{autodoc2-docstring} src.pipeline.features.train.hpo._run_hpo_via_zenml
```
````
