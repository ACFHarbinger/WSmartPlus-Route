# {py:mod}`src.utils.model.loader`

```{py:module} src.utils.model.loader
```

```{autodoc2-docstring} src.utils.model.loader
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`load_model <src.utils.model.loader.load_model>`
  - ```{autodoc2-docstring} src.utils.model.loader.load_model
    :summary:
    ```
* - {py:obj}`_find_latest_checkpoint <src.utils.model.loader._find_latest_checkpoint>`
  - ```{autodoc2-docstring} src.utils.model.loader._find_latest_checkpoint
    :summary:
    ```
* - {py:obj}`_parse_hydra_config <src.utils.model.loader._parse_hydra_config>`
  - ```{autodoc2-docstring} src.utils.model.loader._parse_hydra_config
    :summary:
    ```
* - {py:obj}`_load_hyperparameters <src.utils.model.loader._load_hyperparameters>`
  - ```{autodoc2-docstring} src.utils.model.loader._load_hyperparameters
    :summary:
    ```
* - {py:obj}`_migrate_and_load_state_dict <src.utils.model.loader._migrate_and_load_state_dict>`
  - ```{autodoc2-docstring} src.utils.model.loader._migrate_and_load_state_dict
    :summary:
    ```
````

### API

````{py:function} load_model(path: str, epoch: typing.Optional[int] = None) -> typing.Tuple[torch.nn.Module, typing.Dict[str, typing.Any]]
:canonical: src.utils.model.loader.load_model

```{autodoc2-docstring} src.utils.model.loader.load_model
```
````

````{py:function} _find_latest_checkpoint(path: str, epoch: typing.Optional[int]) -> str
:canonical: src.utils.model.loader._find_latest_checkpoint

```{autodoc2-docstring} src.utils.model.loader._find_latest_checkpoint
```
````

````{py:function} _parse_hydra_config(cfg: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.utils.model.loader._parse_hydra_config

```{autodoc2-docstring} src.utils.model.loader._parse_hydra_config
```
````

````{py:function} _load_hyperparameters(path: str) -> typing.Dict[str, typing.Any]
:canonical: src.utils.model.loader._load_hyperparameters

```{autodoc2-docstring} src.utils.model.loader._load_hyperparameters
```
````

````{py:function} _migrate_and_load_state_dict(model: torch.nn.Module, loaded_state_dict: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.model.loader._migrate_and_load_state_dict

```{autodoc2-docstring} src.utils.model.loader._migrate_and_load_state_dict
```
````
