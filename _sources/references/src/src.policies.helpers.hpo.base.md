# {py:mod}`src.policies.helpers.hpo.base`

```{py:module} src.policies.helpers.hpo.base
```

```{autodoc2-docstring} src.policies.helpers.hpo.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyHPOBase <src.policies.helpers.hpo.base.PolicyHPOBase>`
  - ```{autodoc2-docstring} src.policies.helpers.hpo.base.PolicyHPOBase
    :summary:
    ```
````

### API

`````{py:class} PolicyHPOBase(cfg: logic.src.configs.Config)
:canonical: src.policies.helpers.hpo.base.PolicyHPOBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.helpers.hpo.base.PolicyHPOBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.hpo.base.PolicyHPOBase.__init__
```

````{py:method} run(n_trials: int = 20) -> float
:canonical: src.policies.helpers.hpo.base.PolicyHPOBase.run
:abstractmethod:

```{autodoc2-docstring} src.policies.helpers.hpo.base.PolicyHPOBase.run
```

````

````{py:method} _apply_params(cfg: logic.src.configs.Config, params: typing.Dict[str, typing.Any]) -> None
:canonical: src.policies.helpers.hpo.base.PolicyHPOBase._apply_params

```{autodoc2-docstring} src.policies.helpers.hpo.base.PolicyHPOBase._apply_params
```

````

````{py:method} suggest_param(trial: optuna.Trial, name: str, spec: typing.Dict[str, typing.Any]) -> typing.Any
:canonical: src.policies.helpers.hpo.base.PolicyHPOBase.suggest_param
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.hpo.base.PolicyHPOBase.suggest_param
```

````

`````
