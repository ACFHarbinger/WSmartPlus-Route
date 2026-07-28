# {py:mod}`src.pipeline.rl.hpo.base`

```{py:module} src.pipeline.rl.hpo.base
```

```{autodoc2-docstring} src.pipeline.rl.hpo.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseHPO <src.pipeline.rl.hpo.base.BaseHPO>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.base.BaseHPO
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`normalise_search_space <src.pipeline.rl.hpo.base.normalise_search_space>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.base.normalise_search_space
    :summary:
    ```
* - {py:obj}`apply_params <src.pipeline.rl.hpo.base.apply_params>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.base.apply_params
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ParamSpec <src.pipeline.rl.hpo.base.ParamSpec>`
  - ```{autodoc2-docstring} src.pipeline.rl.hpo.base.ParamSpec
    :summary:
    ```
````

### API

````{py:data} ParamSpec
:canonical: src.pipeline.rl.hpo.base.ParamSpec
:value: >
   None

```{autodoc2-docstring} src.pipeline.rl.hpo.base.ParamSpec
```

````

````{py:function} normalise_search_space(raw: typing.Dict[str, typing.Any]) -> typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec]
:canonical: src.pipeline.rl.hpo.base.normalise_search_space

```{autodoc2-docstring} src.pipeline.rl.hpo.base.normalise_search_space
```
````

````{py:function} apply_params(cfg: logic.src.configs.Config, params: typing.Dict[str, typing.Any]) -> logic.src.configs.Config
:canonical: src.pipeline.rl.hpo.base.apply_params

```{autodoc2-docstring} src.pipeline.rl.hpo.base.apply_params
```
````

`````{py:class} BaseHPO(cfg: logic.src.configs.Config, objective_fn: typing.Callable, search_space: typing.Optional[typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec]] = None)
:canonical: src.pipeline.rl.hpo.base.BaseHPO

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.rl.hpo.base.BaseHPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.hpo.base.BaseHPO.__init__
```

````{py:method} run() -> float
:canonical: src.pipeline.rl.hpo.base.BaseHPO.run
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.hpo.base.BaseHPO.run
```

````

````{py:method} suggest_param_optuna(trial: typing.Any, name: str, spec: src.pipeline.rl.hpo.base.ParamSpec) -> typing.Any
:canonical: src.pipeline.rl.hpo.base.BaseHPO.suggest_param_optuna
:staticmethod:

```{autodoc2-docstring} src.pipeline.rl.hpo.base.BaseHPO.suggest_param_optuna
```

````

````{py:method} build_configspace(search_space: typing.Dict[str, src.pipeline.rl.hpo.base.ParamSpec]) -> typing.Any
:canonical: src.pipeline.rl.hpo.base.BaseHPO.build_configspace
:staticmethod:

```{autodoc2-docstring} src.pipeline.rl.hpo.base.BaseHPO.build_configspace
```

````

`````
