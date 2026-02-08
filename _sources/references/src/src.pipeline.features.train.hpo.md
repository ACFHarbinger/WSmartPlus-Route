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
* - {py:obj}`run_hpo <src.pipeline.features.train.hpo.run_hpo>`
  - ```{autodoc2-docstring} src.pipeline.features.train.hpo.run_hpo
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
````

### API

````{py:data} logger
:canonical: src.pipeline.features.train.hpo.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.train.hpo.logger
```

````

````{py:function} objective(trial: optuna.Trial, base_cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.hpo.objective

```{autodoc2-docstring} src.pipeline.features.train.hpo.objective
```
````

````{py:function} run_hpo(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.hpo.run_hpo

```{autodoc2-docstring} src.pipeline.features.train.hpo.run_hpo
```
````
