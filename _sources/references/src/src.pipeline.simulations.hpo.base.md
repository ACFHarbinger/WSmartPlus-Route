# {py:mod}`src.pipeline.simulations.hpo.base`

```{py:module} src.pipeline.simulations.hpo.base
```

```{autodoc2-docstring} src.pipeline.simulations.hpo.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyHPOBase <src.pipeline.simulations.hpo.base.PolicyHPOBase>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_build_grid_from_search_space <src.pipeline.simulations.hpo.base._build_grid_from_search_space>`
  - ```{autodoc2-docstring} src.pipeline.simulations.hpo.base._build_grid_from_search_space
    :summary:
    ```
````

### API

`````{py:class} PolicyHPOBase(cfg: logic.src.configs.Config, search_space: typing.Optional[typing.Dict[str, typing.Any]] = None)
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase.__init__
```

````{py:method} run(cfg: logic.src.configs.Config) -> typing.Union[float, typing.List[float]]
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase.run
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase.run
```

````

````{py:method} run_iterative(cfg: logic.src.configs.Config, max_steps: int) -> typing.Any
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase.run_iterative

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase.run_iterative
```

````

````{py:method} suggest_params(trial: optuna.Trial) -> typing.Dict[str, typing.Any]
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase.suggest_params

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase.suggest_params
```

````

````{py:method} _apply_params(cfg: logic.src.configs.Config, params: typing.Dict[str, typing.Any]) -> None
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase._apply_params

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase._apply_params
```

````

````{py:method} validate_search_space(space: typing.Dict[str, typing.Any], policy_name: str) -> None
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase.validate_search_space
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase.validate_search_space
```

````

````{py:method} suggest_param(trial: optuna.Trial, name: str, spec: typing.Dict[str, typing.Any]) -> typing.Any
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase.suggest_param
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase.suggest_param
```

````

````{py:method} build_sampler(method: str, seed: int, search_space: typing.Optional[typing.Dict[str, typing.Any]] = None) -> optuna.samplers.BaseSampler
:canonical: src.pipeline.simulations.hpo.base.PolicyHPOBase.build_sampler
:staticmethod:

```{autodoc2-docstring} src.pipeline.simulations.hpo.base.PolicyHPOBase.build_sampler
```

````

`````

````{py:function} _build_grid_from_search_space(search_space: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.List[typing.Any]]
:canonical: src.pipeline.simulations.hpo.base._build_grid_from_search_space

```{autodoc2-docstring} src.pipeline.simulations.hpo.base._build_grid_from_search_space
```
````
