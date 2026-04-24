# {py:mod}`src.policies.helpers.hpo.hpo_handler`

```{py:module} src.policies.helpers.hpo.hpo_handler
```

```{autodoc2-docstring} src.policies.helpers.hpo.hpo_handler
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`objective <src.policies.helpers.hpo.hpo_handler.objective>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.hpo_handler.objective
    :summary:
    ```
* - {py:obj}`run_hpo_sim <src.policies.helpers.hpo.hpo_handler.run_hpo_sim>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.hpo_handler.run_hpo_sim
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.hpo.hpo_handler.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.hpo_handler.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.hpo.hpo_handler.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.policies.helpers.hpo.hpo_handler.logger
```

````

````{py:function} objective(trial: optuna.Trial, base_cfg: logic.src.configs.Config, data_size: int, lock: typing.Any) -> float
:canonical: src.policies.helpers.hpo.hpo_handler.objective

```{autodoc2-docstring} src.policies.helpers.hpo.hpo_handler.objective
```
````

````{py:function} run_hpo_sim(cfg: logic.src.configs.Config) -> float
:canonical: src.policies.helpers.hpo.hpo_handler.run_hpo_sim

```{autodoc2-docstring} src.policies.helpers.hpo.hpo_handler.run_hpo_sim
```
````
