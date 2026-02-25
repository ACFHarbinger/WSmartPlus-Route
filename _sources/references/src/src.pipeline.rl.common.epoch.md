# {py:mod}`src.pipeline.rl.common.epoch`

```{py:module} src.pipeline.rl.common.epoch
```

```{autodoc2-docstring} src.pipeline.rl.common.epoch
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`prepare_epoch <src.pipeline.rl.common.epoch.prepare_epoch>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch.prepare_epoch
    :summary:
    ```
* - {py:obj}`regenerate_dataset <src.pipeline.rl.common.epoch.regenerate_dataset>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch.regenerate_dataset
    :summary:
    ```
* - {py:obj}`compute_validation_metrics <src.pipeline.rl.common.epoch.compute_validation_metrics>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch.compute_validation_metrics
    :summary:
    ```
* - {py:obj}`_add_reward_metric <src.pipeline.rl.common.epoch._add_reward_metric>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_reward_metric
    :summary:
    ```
* - {py:obj}`_add_costs_metrics <src.pipeline.rl.common.epoch._add_costs_metrics>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_costs_metrics
    :summary:
    ```
* - {py:obj}`_add_efficiency_metric <src.pipeline.rl.common.epoch._add_efficiency_metric>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_efficiency_metric
    :summary:
    ```
* - {py:obj}`_add_overflow_metrics <src.pipeline.rl.common.epoch._add_overflow_metrics>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_overflow_metrics
    :summary:
    ```
* - {py:obj}`_build_visited_mask <src.pipeline.rl.common.epoch._build_visited_mask>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch._build_visited_mask
    :summary:
    ```
* - {py:obj}`_get_next_day_waste <src.pipeline.rl.common.epoch._get_next_day_waste>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch._get_next_day_waste
    :summary:
    ```
* - {py:obj}`apply_time_step <src.pipeline.rl.common.epoch.apply_time_step>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch.apply_time_step
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.rl.common.epoch.logger>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.epoch.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.rl.common.epoch.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.rl.common.epoch.logger
```

````

````{py:function} prepare_epoch(model: torch.nn.Module, env: typing.Any, baseline: typing.Any, dataset: torch.utils.data.Dataset, epoch: int, phase: str = 'train') -> torch.utils.data.Dataset
:canonical: src.pipeline.rl.common.epoch.prepare_epoch

```{autodoc2-docstring} src.pipeline.rl.common.epoch.prepare_epoch
```
````

````{py:function} regenerate_dataset(env: typing.Any, size: int) -> typing.Optional[torch.utils.data.Dataset]
:canonical: src.pipeline.rl.common.epoch.regenerate_dataset

```{autodoc2-docstring} src.pipeline.rl.common.epoch.regenerate_dataset
```
````

````{py:function} compute_validation_metrics(out: typing.Dict, batch: tensordict.TensorDict, env: typing.Any) -> typing.Dict[str, float]
:canonical: src.pipeline.rl.common.epoch.compute_validation_metrics

```{autodoc2-docstring} src.pipeline.rl.common.epoch.compute_validation_metrics
```
````

````{py:function} _add_reward_metric(metrics: typing.Dict[str, float], out: typing.Dict) -> None
:canonical: src.pipeline.rl.common.epoch._add_reward_metric

```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_reward_metric
```
````

````{py:function} _add_costs_metrics(metrics: typing.Dict[str, float], out: typing.Dict, batch: tensordict.TensorDict, env: typing.Any) -> None
:canonical: src.pipeline.rl.common.epoch._add_costs_metrics

```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_costs_metrics
```
````

````{py:function} _add_efficiency_metric(metrics: typing.Dict[str, float]) -> None
:canonical: src.pipeline.rl.common.epoch._add_efficiency_metric

```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_efficiency_metric
```
````

````{py:function} _add_overflow_metrics(metrics: typing.Dict[str, float], out: typing.Dict, batch: tensordict.TensorDict, env: typing.Any) -> None
:canonical: src.pipeline.rl.common.epoch._add_overflow_metrics

```{autodoc2-docstring} src.pipeline.rl.common.epoch._add_overflow_metrics
```
````

````{py:function} _build_visited_mask(epoch_actions: typing.List[torch.Tensor], batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor
:canonical: src.pipeline.rl.common.epoch._build_visited_mask

```{autodoc2-docstring} src.pipeline.rl.common.epoch._build_visited_mask
```
````

````{py:function} _get_next_day_waste(td: tensordict.TensorDict, current_fill: torch.Tensor, day: int, env: typing.Any, batch_size: int, device: torch.device, key: str = 'waste') -> torch.Tensor
:canonical: src.pipeline.rl.common.epoch._get_next_day_waste

```{autodoc2-docstring} src.pipeline.rl.common.epoch._get_next_day_waste
```
````

````{py:function} apply_time_step(dataset: torch.utils.data.Dataset, epoch_actions: typing.List[torch.Tensor], day: int, env: typing.Any) -> torch.utils.data.Dataset
:canonical: src.pipeline.rl.common.epoch.apply_time_step

```{autodoc2-docstring} src.pipeline.rl.common.epoch.apply_time_step
```
````
