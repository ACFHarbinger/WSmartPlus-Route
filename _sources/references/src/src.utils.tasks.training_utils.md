# {py:mod}`src.utils.tasks.training_utils`

```{py:module} src.utils.tasks.training_utils
```

```{autodoc2-docstring} src.utils.tasks.training_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_optimizer <src.utils.tasks.training_utils.get_optimizer>`
  - ```{autodoc2-docstring} src.utils.tasks.training_utils.get_optimizer
    :summary:
    ```
* - {py:obj}`get_scheduler <src.utils.tasks.training_utils.get_scheduler>`
  - ```{autodoc2-docstring} src.utils.tasks.training_utils.get_scheduler
    :summary:
    ```
* - {py:obj}`get_lightning_device <src.utils.tasks.training_utils.get_lightning_device>`
  - ```{autodoc2-docstring} src.utils.tasks.training_utils.get_lightning_device
    :summary:
    ```
````

### API

````{py:function} get_optimizer(name: str, parameters: typing.Any, lr: float = 0.0001, weight_decay: float = 0.0, **kwargs: typing.Any) -> torch.optim.Optimizer
:canonical: src.utils.tasks.training_utils.get_optimizer

```{autodoc2-docstring} src.utils.tasks.training_utils.get_optimizer
```
````

````{py:function} get_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs: typing.Any) -> typing.Optional[typing.Any]
:canonical: src.utils.tasks.training_utils.get_scheduler

```{autodoc2-docstring} src.utils.tasks.training_utils.get_scheduler
```
````

````{py:function} get_lightning_device(trainer: typing.Any) -> torch.device
:canonical: src.utils.tasks.training_utils.get_lightning_device

```{autodoc2-docstring} src.utils.tasks.training_utils.get_lightning_device
```
````
