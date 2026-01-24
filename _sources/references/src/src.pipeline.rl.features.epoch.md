# {py:mod}`src.pipeline.rl.features.epoch`

```{py:module} src.pipeline.rl.features.epoch
```

```{autodoc2-docstring} src.pipeline.rl.features.epoch
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`prepare_epoch <src.pipeline.rl.features.epoch.prepare_epoch>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.epoch.prepare_epoch
    :summary:
    ```
* - {py:obj}`regenerate_dataset <src.pipeline.rl.features.epoch.regenerate_dataset>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.epoch.regenerate_dataset
    :summary:
    ```
* - {py:obj}`compute_validation_metrics <src.pipeline.rl.features.epoch.compute_validation_metrics>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.epoch.compute_validation_metrics
    :summary:
    ```
````

### API

````{py:function} prepare_epoch(model: torch.nn.Module, env: any, baseline: any, dataset: torch.utils.data.Dataset, epoch: int, phase: str = 'train') -> torch.utils.data.Dataset
:canonical: src.pipeline.rl.features.epoch.prepare_epoch

```{autodoc2-docstring} src.pipeline.rl.features.epoch.prepare_epoch
```
````

````{py:function} regenerate_dataset(env: any, size: int) -> typing.Optional[torch.utils.data.Dataset]
:canonical: src.pipeline.rl.features.epoch.regenerate_dataset

```{autodoc2-docstring} src.pipeline.rl.features.epoch.regenerate_dataset
```
````

````{py:function} compute_validation_metrics(out: typing.Dict, batch: tensordict.TensorDict, env: any) -> typing.Dict[str, float]
:canonical: src.pipeline.rl.features.epoch.compute_validation_metrics

```{autodoc2-docstring} src.pipeline.rl.features.epoch.compute_validation_metrics
```
````
