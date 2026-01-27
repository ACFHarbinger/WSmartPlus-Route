# {py:mod}`src.pipeline.features.train`

```{py:module} src.pipeline.features.train
```

```{autodoc2-docstring} src.pipeline.features.train
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_model <src.pipeline.features.train.create_model>`
  - ```{autodoc2-docstring} src.pipeline.features.train.create_model
    :summary:
    ```
* - {py:obj}`objective <src.pipeline.features.train.objective>`
  - ```{autodoc2-docstring} src.pipeline.features.train.objective
    :summary:
    ```
* - {py:obj}`run_hpo <src.pipeline.features.train.run_hpo>`
  - ```{autodoc2-docstring} src.pipeline.features.train.run_hpo
    :summary:
    ```
* - {py:obj}`run_training <src.pipeline.features.train.run_training>`
  - ```{autodoc2-docstring} src.pipeline.features.train.run_training
    :summary:
    ```
* - {py:obj}`main <src.pipeline.features.train.main>`
  - ```{autodoc2-docstring} src.pipeline.features.train.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.features.train.logger>`
  - ```{autodoc2-docstring} src.pipeline.features.train.logger
    :summary:
    ```
* - {py:obj}`cs <src.pipeline.features.train.cs>`
  - ```{autodoc2-docstring} src.pipeline.features.train.cs
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.features.train.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.train.logger
```

````

````{py:data} cs
:canonical: src.pipeline.features.train.cs
:value: >
   'instance(...)'

```{autodoc2-docstring} src.pipeline.features.train.cs
```

````

````{py:function} create_model(cfg: logic.src.configs.Config) -> pytorch_lightning.LightningModule
:canonical: src.pipeline.features.train.create_model

```{autodoc2-docstring} src.pipeline.features.train.create_model
```
````

````{py:function} objective(trial: optuna.Trial, base_cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.objective

```{autodoc2-docstring} src.pipeline.features.train.objective
```
````

````{py:function} run_hpo(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.run_hpo

```{autodoc2-docstring} src.pipeline.features.train.run_hpo
```
````

````{py:function} run_training(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.run_training

```{autodoc2-docstring} src.pipeline.features.train.run_training
```
````

````{py:function} main(cfg: logic.src.configs.Config) -> float
:canonical: src.pipeline.features.train.main

```{autodoc2-docstring} src.pipeline.features.train.main
```
````
