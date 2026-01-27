# {py:mod}`src.pipeline.rl.hpo.optuna_hpo`

```{py:module} src.pipeline.rl.hpo.optuna_hpo
```

```{autodoc2-docstring} src.pipeline.rl.hpo.optuna_hpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OptunaHPO <src.pipeline.rl.hpo.optuna_hpo.OptunaHPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.optuna_hpo.OptunaHPO
    :summary:
    ```
````

### API

`````{py:class} OptunaHPO(cfg: logic.src.configs.Config, objective_fn: typing.Callable)
:canonical: src.pipeline.rl.hpo.optuna_hpo.OptunaHPO

```{autodoc2-docstring} src.pipeline.rl.hpo.optuna_hpo.OptunaHPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.optuna_hpo.OptunaHPO.__init__
```

````{py:method} run() -> float
:canonical: src.pipeline.rl.hpo.optuna_hpo.OptunaHPO.run

```{autodoc2-docstring} src.pipeline.rl.hpo.optuna_hpo.OptunaHPO.run
```

````

````{py:method} _get_sampler() -> optuna.samplers.BaseSampler
:canonical: src.pipeline.rl.hpo.optuna_hpo.OptunaHPO._get_sampler

```{autodoc2-docstring} src.pipeline.rl.hpo.optuna_hpo.OptunaHPO._get_sampler
```

````

````{py:method} _get_pruner() -> optuna.pruners.BasePruner
:canonical: src.pipeline.rl.hpo.optuna_hpo.OptunaHPO._get_pruner

```{autodoc2-docstring} src.pipeline.rl.hpo.optuna_hpo.OptunaHPO._get_pruner
```

````

`````
