# {py:mod}`src.utils.logging.modules.metrics`

```{py:module} src.utils.logging.modules.metrics
```

```{autodoc2-docstring} src.utils.logging.modules.metrics
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`log_values <src.utils.logging.modules.metrics.log_values>`
  - ```{autodoc2-docstring} src.utils.logging.modules.metrics.log_values
    :summary:
    ```
* - {py:obj}`log_epoch <src.utils.logging.modules.metrics.log_epoch>`
  - ```{autodoc2-docstring} src.utils.logging.modules.metrics.log_epoch
    :summary:
    ```
* - {py:obj}`get_loss_stats <src.utils.logging.modules.metrics.get_loss_stats>`
  - ```{autodoc2-docstring} src.utils.logging.modules.metrics.get_loss_stats
    :summary:
    ```
````

### API

````{py:function} log_values(cost: torch.Tensor, grad_norms: typing.Tuple[torch.Tensor, ...], epoch: int, batch_id: int, step: int, l_dict: typing.Dict[str, torch.Tensor], tb_logger: typing.Any, opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.logging.modules.metrics.log_values

```{autodoc2-docstring} src.utils.logging.modules.metrics.log_values
```
````

````{py:function} log_epoch(x_tup: typing.Tuple[str, int], loss_keys: typing.List[str], epoch_loss: typing.Dict[str, typing.List[torch.Tensor]], opts: typing.Dict[str, typing.Any]) -> None
:canonical: src.utils.logging.modules.metrics.log_epoch

```{autodoc2-docstring} src.utils.logging.modules.metrics.log_epoch
```
````

````{py:function} get_loss_stats(epoch_loss: typing.Dict[str, typing.List[torch.Tensor]]) -> typing.List[float]
:canonical: src.utils.logging.modules.metrics.get_loss_stats

```{autodoc2-docstring} src.utils.logging.modules.metrics.get_loss_stats
```
````
