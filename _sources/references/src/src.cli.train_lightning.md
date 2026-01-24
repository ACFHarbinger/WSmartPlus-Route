# {py:mod}`src.cli.train_lightning`

```{py:module} src.cli.train_lightning
```

```{autodoc2-docstring} src.cli.train_lightning
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_model <src.cli.train_lightning.create_model>`
  - ```{autodoc2-docstring} src.cli.train_lightning.create_model
    :summary:
    ```
* - {py:obj}`objective <src.cli.train_lightning.objective>`
  - ```{autodoc2-docstring} src.cli.train_lightning.objective
    :summary:
    ```
* - {py:obj}`run_hpo <src.cli.train_lightning.run_hpo>`
  - ```{autodoc2-docstring} src.cli.train_lightning.run_hpo
    :summary:
    ```
* - {py:obj}`run_training <src.cli.train_lightning.run_training>`
  - ```{autodoc2-docstring} src.cli.train_lightning.run_training
    :summary:
    ```
* - {py:obj}`main <src.cli.train_lightning.main>`
  - ```{autodoc2-docstring} src.cli.train_lightning.main
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.cli.train_lightning.logger>`
  - ```{autodoc2-docstring} src.cli.train_lightning.logger
    :summary:
    ```
* - {py:obj}`cs <src.cli.train_lightning.cs>`
  - ```{autodoc2-docstring} src.cli.train_lightning.cs
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.cli.train_lightning.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.cli.train_lightning.logger
```

````

````{py:data} cs
:canonical: src.cli.train_lightning.cs
:value: >
   'instance(...)'

```{autodoc2-docstring} src.cli.train_lightning.cs
```

````

````{py:function} create_model(cfg: logic.src.configs.Config) -> pytorch_lightning.LightningModule
:canonical: src.cli.train_lightning.create_model

```{autodoc2-docstring} src.cli.train_lightning.create_model
```
````

````{py:function} objective(trial: optuna.Trial, base_cfg: logic.src.configs.Config) -> float
:canonical: src.cli.train_lightning.objective

```{autodoc2-docstring} src.cli.train_lightning.objective
```
````

````{py:function} run_hpo(cfg: logic.src.configs.Config) -> float
:canonical: src.cli.train_lightning.run_hpo

```{autodoc2-docstring} src.cli.train_lightning.run_hpo
```
````

````{py:function} run_training(cfg: logic.src.configs.Config) -> float
:canonical: src.cli.train_lightning.run_training

```{autodoc2-docstring} src.cli.train_lightning.run_training
```
````

````{py:function} main(cfg: logic.src.configs.Config) -> float
:canonical: src.cli.train_lightning.main

```{autodoc2-docstring} src.cli.train_lightning.main
```
````
