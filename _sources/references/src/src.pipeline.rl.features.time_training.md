# {py:mod}`src.pipeline.rl.features.time_training`

```{py:module} src.pipeline.rl.features.time_training
```

```{autodoc2-docstring} src.pipeline.rl.features.time_training
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TimeBasedMixin <src.pipeline.rl.features.time_training.TimeBasedMixin>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.time_training.TimeBasedMixin
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`prepare_time_dataset <src.pipeline.rl.features.time_training.prepare_time_dataset>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.time_training.prepare_time_dataset
    :summary:
    ```
````

### API

`````{py:class} TimeBasedMixin
:canonical: src.pipeline.rl.features.time_training.TimeBasedMixin

```{autodoc2-docstring} src.pipeline.rl.features.time_training.TimeBasedMixin
```

````{py:method} setup_time_training(opts: typing.Dict)
:canonical: src.pipeline.rl.features.time_training.TimeBasedMixin.setup_time_training

```{autodoc2-docstring} src.pipeline.rl.features.time_training.TimeBasedMixin.setup_time_training
```

````

````{py:method} update_dataset_for_day(routes: typing.List[torch.Tensor], day: int)
:canonical: src.pipeline.rl.features.time_training.TimeBasedMixin.update_dataset_for_day

```{autodoc2-docstring} src.pipeline.rl.features.time_training.TimeBasedMixin.update_dataset_for_day
```

````

`````

````{py:function} prepare_time_dataset(dataset, day, history)
:canonical: src.pipeline.rl.features.time_training.prepare_time_dataset

```{autodoc2-docstring} src.pipeline.rl.features.time_training.prepare_time_dataset
```
````
