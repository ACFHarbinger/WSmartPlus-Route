# {py:mod}`src.configs.tasks.train`

```{py:module} src.configs.tasks.train
```

```{autodoc2-docstring} src.configs.tasks.train
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrainConfig <src.configs.tasks.train.TrainConfig>`
  - ```{autodoc2-docstring} src.configs.tasks.train.TrainConfig
    :summary:
    ```
````

### API

`````{py:class} TrainConfig
:canonical: src.configs.tasks.train.TrainConfig

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig
```

````{py:attribute} n_epochs
:canonical: src.configs.tasks.train.TrainConfig.n_epochs
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.n_epochs
```

````

````{py:attribute} batch_size
:canonical: src.configs.tasks.train.TrainConfig.batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.batch_size
```

````

````{py:attribute} train_data_size
:canonical: src.configs.tasks.train.TrainConfig.train_data_size
:type: int
:value: >
   100000

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.train_data_size
```

````

````{py:attribute} val_data_size
:canonical: src.configs.tasks.train.TrainConfig.val_data_size
:type: int
:value: >
   10000

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.val_data_size
```

````

````{py:attribute} val_dataset
:canonical: src.configs.tasks.train.TrainConfig.val_dataset
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.val_dataset
```

````

````{py:attribute} train_dataset
:canonical: src.configs.tasks.train.TrainConfig.train_dataset
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.train_dataset
```

````

````{py:attribute} load_dataset
:canonical: src.configs.tasks.train.TrainConfig.load_dataset
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.load_dataset
```

````

````{py:attribute} num_workers
:canonical: src.configs.tasks.train.TrainConfig.num_workers
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.num_workers
```

````

````{py:attribute} data_distribution
:canonical: src.configs.tasks.train.TrainConfig.data_distribution
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.data_distribution
```

````

````{py:attribute} precision
:canonical: src.configs.tasks.train.TrainConfig.precision
:type: str
:value: >
   '16-mixed'

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.precision
```

````

````{py:attribute} train_time
:canonical: src.configs.tasks.train.TrainConfig.train_time
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.train_time
```

````

````{py:attribute} eval_time_days
:canonical: src.configs.tasks.train.TrainConfig.eval_time_days
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.eval_time_days
```

````

````{py:attribute} accumulation_steps
:canonical: src.configs.tasks.train.TrainConfig.accumulation_steps
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.accumulation_steps
```

````

````{py:attribute} enable_scaler
:canonical: src.configs.tasks.train.TrainConfig.enable_scaler
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.enable_scaler
```

````

````{py:attribute} checkpoint_epochs
:canonical: src.configs.tasks.train.TrainConfig.checkpoint_epochs
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.checkpoint_epochs
```

````

````{py:attribute} shrink_size
:canonical: src.configs.tasks.train.TrainConfig.shrink_size
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.shrink_size
```

````

````{py:attribute} post_processing_epochs
:canonical: src.configs.tasks.train.TrainConfig.post_processing_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.post_processing_epochs
```

````

````{py:attribute} lr_post_processing
:canonical: src.configs.tasks.train.TrainConfig.lr_post_processing
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.lr_post_processing
```

````

````{py:attribute} efficiency_weight
:canonical: src.configs.tasks.train.TrainConfig.efficiency_weight
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.efficiency_weight
```

````

````{py:attribute} overflow_weight
:canonical: src.configs.tasks.train.TrainConfig.overflow_weight
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.overflow_weight
```

````

````{py:attribute} log_step
:canonical: src.configs.tasks.train.TrainConfig.log_step
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.log_step
```

````

````{py:attribute} epoch_start
:canonical: src.configs.tasks.train.TrainConfig.epoch_start
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.epoch_start
```

````

````{py:attribute} eval_only
:canonical: src.configs.tasks.train.TrainConfig.eval_only
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.eval_only
```

````

````{py:attribute} checkpoint_encoder
:canonical: src.configs.tasks.train.TrainConfig.checkpoint_encoder
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.checkpoint_encoder
```

````

````{py:attribute} resume
:canonical: src.configs.tasks.train.TrainConfig.resume
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.resume
```

````

````{py:attribute} logs_dir
:canonical: src.configs.tasks.train.TrainConfig.logs_dir
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.logs_dir
```

````

````{py:attribute} model_weights_path
:canonical: src.configs.tasks.train.TrainConfig.model_weights_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.model_weights_path
```

````

````{py:attribute} final_model_path
:canonical: src.configs.tasks.train.TrainConfig.final_model_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.final_model_path
```

````

````{py:attribute} eval_batch_size
:canonical: src.configs.tasks.train.TrainConfig.eval_batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.eval_batch_size
```

````

````{py:attribute} persistent_workers
:canonical: src.configs.tasks.train.TrainConfig.persistent_workers
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.persistent_workers
```

````

````{py:attribute} pin_memory
:canonical: src.configs.tasks.train.TrainConfig.pin_memory
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.pin_memory
```

````

````{py:attribute} reload_dataloaders_every_n_epochs
:canonical: src.configs.tasks.train.TrainConfig.reload_dataloaders_every_n_epochs
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.reload_dataloaders_every_n_epochs
```

````

````{py:attribute} devices
:canonical: src.configs.tasks.train.TrainConfig.devices
:type: typing.Union[int, str]
:value: >
   'auto'

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.devices
```

````

````{py:attribute} strategy
:canonical: src.configs.tasks.train.TrainConfig.strategy
:type: typing.Optional[str]
:value: >
   'auto'

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.strategy
```

````

````{py:attribute} graph
:canonical: src.configs.tasks.train.TrainConfig.graph
:type: src.configs.envs.graph.GraphConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.graph
```

````

````{py:attribute} reward
:canonical: src.configs.tasks.train.TrainConfig.reward
:type: src.configs.envs.objective.ObjectiveConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.reward
```

````

````{py:attribute} decoding
:canonical: src.configs.tasks.train.TrainConfig.decoding
:type: src.configs.models.decoding.DecodingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.decoding
```

````

````{py:attribute} policy
:canonical: src.configs.tasks.train.TrainConfig.policy
:type: src.configs.policies.neural.NeuralConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.policy
```

````

````{py:attribute} callbacks
:canonical: src.configs.tasks.train.TrainConfig.callbacks
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.configs.tasks.train.TrainConfig.callbacks
```

````

`````
