# {py:mod}`src.utils.configs.setup_manager`

```{py:module} src.utils.configs.setup_manager
```

```{autodoc2-docstring} src.utils.configs.setup_manager
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_cfg_attr <src.utils.configs.setup_manager._get_cfg_attr>`
  - ```{autodoc2-docstring} src.utils.configs.setup_manager._get_cfg_attr
    :summary:
    ```
* - {py:obj}`_resolve_hrl_path <src.utils.configs.setup_manager._resolve_hrl_path>`
  - ```{autodoc2-docstring} src.utils.configs.setup_manager._resolve_hrl_path
    :summary:
    ```
* - {py:obj}`_resolve_checkpoint_file <src.utils.configs.setup_manager._resolve_checkpoint_file>`
  - ```{autodoc2-docstring} src.utils.configs.setup_manager._resolve_checkpoint_file
    :summary:
    ```
* - {py:obj}`_resolve_param <src.utils.configs.setup_manager._resolve_param>`
  - ```{autodoc2-docstring} src.utils.configs.setup_manager._resolve_param
    :summary:
    ```
* - {py:obj}`setup_hrl_manager <src.utils.configs.setup_manager.setup_hrl_manager>`
  - ```{autodoc2-docstring} src.utils.configs.setup_manager.setup_hrl_manager
    :summary:
    ```
````

### API

````{py:function} _get_cfg_attr(sim_cfg: typing.Any, name: str, default: typing.Any = None) -> typing.Any
:canonical: src.utils.configs.setup_manager._get_cfg_attr

```{autodoc2-docstring} src.utils.configs.setup_manager._get_cfg_attr
```
````

````{py:function} _resolve_hrl_path(model_paths: typing.Dict[str, str], policy: str) -> typing.Optional[str]
:canonical: src.utils.configs.setup_manager._resolve_hrl_path

```{autodoc2-docstring} src.utils.configs.setup_manager._resolve_hrl_path
```
````

````{py:function} _resolve_checkpoint_file(hrl_path: str, base_path: typing.Optional[str]) -> typing.Optional[str]
:canonical: src.utils.configs.setup_manager._resolve_checkpoint_file

```{autodoc2-docstring} src.utils.configs.setup_manager._resolve_checkpoint_file
```
````

````{py:function} _resolve_param(configs: typing.Dict[str, typing.Any], sim_cfg: typing.Any, name: str, default: typing.Any) -> typing.Any
:canonical: src.utils.configs.setup_manager._resolve_param

```{autodoc2-docstring} src.utils.configs.setup_manager._resolve_param
```
````

````{py:function} setup_hrl_manager(sim_cfg: typing.Any, device: torch.device, configs: typing.Optional[typing.Dict[str, typing.Any]] = None, policy: typing.Optional[str] = None, base_path: typing.Optional[str] = None, worker_model: typing.Optional[torch.nn.Module] = None) -> typing.Optional[logic.src.models.MustGoManager]
:canonical: src.utils.configs.setup_manager.setup_hrl_manager

```{autodoc2-docstring} src.utils.configs.setup_manager.setup_hrl_manager
```
````
