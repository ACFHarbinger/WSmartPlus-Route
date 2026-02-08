# {py:mod}`src.utils.configs.setup_env`

```{py:module} src.utils.configs.setup_env
```

```{autodoc2-docstring} src.utils.configs.setup_env
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup_cost_weights <src.utils.configs.setup_env.setup_cost_weights>`
  - ```{autodoc2-docstring} src.utils.configs.setup_env.setup_cost_weights
    :summary:
    ```
* - {py:obj}`setup_env <src.utils.configs.setup_env.setup_env>`
  - ```{autodoc2-docstring} src.utils.configs.setup_env.setup_env
    :summary:
    ```
````

### API

````{py:function} setup_cost_weights(opts: typing.Dict[str, typing.Any], def_val: float = 1.0) -> typing.Dict[str, float]
:canonical: src.utils.configs.setup_env.setup_cost_weights

```{autodoc2-docstring} src.utils.configs.setup_env.setup_cost_weights
```
````

````{py:function} setup_env(policy: str, server: bool = False, gplic_filename: typing.Optional[str] = None, symkey_name: typing.Optional[str] = None, env_filename: typing.Optional[str] = None) -> typing.Optional[gurobipy.Env]
:canonical: src.utils.configs.setup_env.setup_env

```{autodoc2-docstring} src.utils.configs.setup_env.setup_env
```
````
