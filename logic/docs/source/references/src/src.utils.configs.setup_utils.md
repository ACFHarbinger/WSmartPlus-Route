# {py:mod}`src.utils.configs.setup_utils`

```{py:module} src.utils.configs.setup_utils
```

```{autodoc2-docstring} src.utils.configs.setup_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup_cost_weights <src.utils.configs.setup_utils.setup_cost_weights>`
  - ```{autodoc2-docstring} src.utils.configs.setup_utils.setup_cost_weights
    :summary:
    ```
* - {py:obj}`setup_hrl_manager <src.utils.configs.setup_utils.setup_hrl_manager>`
  - ```{autodoc2-docstring} src.utils.configs.setup_utils.setup_hrl_manager
    :summary:
    ```
* - {py:obj}`setup_model <src.utils.configs.setup_utils.setup_model>`
  - ```{autodoc2-docstring} src.utils.configs.setup_utils.setup_model
    :summary:
    ```
* - {py:obj}`setup_env <src.utils.configs.setup_utils.setup_env>`
  - ```{autodoc2-docstring} src.utils.configs.setup_utils.setup_env
    :summary:
    ```
* - {py:obj}`setup_model_and_baseline <src.utils.configs.setup_utils.setup_model_and_baseline>`
  - ```{autodoc2-docstring} src.utils.configs.setup_utils.setup_model_and_baseline
    :summary:
    ```
* - {py:obj}`setup_optimizer_and_lr_scheduler <src.utils.configs.setup_utils.setup_optimizer_and_lr_scheduler>`
  - ```{autodoc2-docstring} src.utils.configs.setup_utils.setup_optimizer_and_lr_scheduler
    :summary:
    ```
````

### API

````{py:function} setup_cost_weights(opts: typing.Dict[str, typing.Any], def_val: float = 1.0) -> typing.Dict[str, float]
:canonical: src.utils.configs.setup_utils.setup_cost_weights

```{autodoc2-docstring} src.utils.configs.setup_utils.setup_cost_weights
```
````

````{py:function} setup_hrl_manager(opts: typing.Dict[str, typing.Any], device: torch.device, configs: typing.Optional[typing.Dict[str, typing.Any]] = None, policy: typing.Optional[str] = None, base_path: typing.Optional[str] = None, worker_model: typing.Optional[torch.nn.Module] = None) -> typing.Optional[logic.src.models.GATLSTManager]
:canonical: src.utils.configs.setup_utils.setup_hrl_manager

```{autodoc2-docstring} src.utils.configs.setup_utils.setup_hrl_manager
```
````

````{py:function} setup_model(policy: str, general_path: str, model_paths: typing.Dict[str, str], device: torch.device, lock: threading.Lock, temperature: float = 1.0, decode_type: str = 'greedy') -> typing.Tuple[torch.nn.Module, typing.Dict[str, typing.Any]]
:canonical: src.utils.configs.setup_utils.setup_model

```{autodoc2-docstring} src.utils.configs.setup_utils.setup_model
```
````

````{py:function} setup_env(policy: str, server: bool = False, gplic_filename: typing.Optional[str] = None, symkey_name: typing.Optional[str] = None, env_filename: typing.Optional[str] = None) -> typing.Optional[gurobipy.Env]
:canonical: src.utils.configs.setup_utils.setup_env

```{autodoc2-docstring} src.utils.configs.setup_utils.setup_env
```
````

````{py:function} setup_model_and_baseline(problem: typing.Any, data_load: typing.Dict[str, typing.Any], use_cuda: bool, opts: typing.Dict[str, typing.Any]) -> typing.Tuple[torch.nn.Module, typing.Any]
:canonical: src.utils.configs.setup_utils.setup_model_and_baseline

```{autodoc2-docstring} src.utils.configs.setup_utils.setup_model_and_baseline
```
````

````{py:function} setup_optimizer_and_lr_scheduler(model: torch.nn.Module, baseline: typing.Any, data_load: typing.Dict[str, typing.Any], opts: typing.Dict[str, typing.Any]) -> typing.Tuple[torch.optim.Optimizer, typing.Any]
:canonical: src.utils.configs.setup_utils.setup_optimizer_and_lr_scheduler

```{autodoc2-docstring} src.utils.configs.setup_utils.setup_optimizer_and_lr_scheduler
```
````
