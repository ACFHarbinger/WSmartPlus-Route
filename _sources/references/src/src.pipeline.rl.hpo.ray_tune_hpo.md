# {py:mod}`src.pipeline.rl.hpo.ray_tune_hpo`

```{py:module} src.pipeline.rl.hpo.ray_tune_hpo
```

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RayTuneHPO <src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO
    :summary:
    ```
````

### API

`````{py:class} RayTuneHPO(cfg: logic.src.configs.Config, objective_fn: typing.Callable, search_space: typing.Optional[typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec]] = None, scheduler: str = 'asha', mlflow_tracking_uri: typing.Optional[str] = None, mlflow_experiment_name: typing.Optional[str] = None)
:canonical: src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO

Bases: {py:obj}`src.pipeline.rl.hpo.base.BaseHPO`

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO.__init__
```

````{py:method} run() -> float
:canonical: src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO.run

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO.run
```

````

````{py:method} _build_ray_search_space() -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_ray_search_space

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_ray_search_space
```

````

````{py:method} _build_scheduler() -> typing.Any
:canonical: src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_scheduler

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_scheduler
```

````

````{py:method} _build_search_alg() -> typing.Optional[typing.Any]
:canonical: src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_search_alg

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_search_alg
```

````

````{py:method} _build_run_callbacks() -> list
:canonical: src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_run_callbacks

```{autodoc2-docstring} src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO._build_run_callbacks
```

````

`````
