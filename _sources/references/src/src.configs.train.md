# {py:mod}`src.configs.train`

```{py:module} src.configs.train
```

```{autodoc2-docstring} src.configs.train
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrainConfig <src.configs.train.TrainConfig>`
  - ```{autodoc2-docstring} src.configs.train.TrainConfig
    :summary:
    ```
````

### API

`````{py:class} TrainConfig
:canonical: src.configs.train.TrainConfig

```{autodoc2-docstring} src.configs.train.TrainConfig
```

````{py:attribute} n_epochs
:canonical: src.configs.train.TrainConfig.n_epochs
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.train.TrainConfig.n_epochs
```

````

````{py:attribute} batch_size
:canonical: src.configs.train.TrainConfig.batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} src.configs.train.TrainConfig.batch_size
```

````

````{py:attribute} train_data_size
:canonical: src.configs.train.TrainConfig.train_data_size
:type: int
:value: >
   100000

```{autodoc2-docstring} src.configs.train.TrainConfig.train_data_size
```

````

````{py:attribute} val_data_size
:canonical: src.configs.train.TrainConfig.val_data_size
:type: int
:value: >
   10000

```{autodoc2-docstring} src.configs.train.TrainConfig.val_data_size
```

````

````{py:attribute} val_dataset
:canonical: src.configs.train.TrainConfig.val_dataset
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.train.TrainConfig.val_dataset
```

````

````{py:attribute} num_workers
:canonical: src.configs.train.TrainConfig.num_workers
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.train.TrainConfig.num_workers
```

````

````{py:attribute} precision
:canonical: src.configs.train.TrainConfig.precision
:type: str
:value: >
   '16-mixed'

```{autodoc2-docstring} src.configs.train.TrainConfig.precision
```

````

````{py:attribute} train_time
:canonical: src.configs.train.TrainConfig.train_time
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.train.TrainConfig.train_time
```

````

````{py:attribute} eval_time_days
:canonical: src.configs.train.TrainConfig.eval_time_days
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.train.TrainConfig.eval_time_days
```

````

````{py:attribute} accumulation_steps
:canonical: src.configs.train.TrainConfig.accumulation_steps
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.train.TrainConfig.accumulation_steps
```

````

````{py:attribute} enable_scaler
:canonical: src.configs.train.TrainConfig.enable_scaler
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.train.TrainConfig.enable_scaler
```

````

````{py:attribute} checkpoint_epochs
:canonical: src.configs.train.TrainConfig.checkpoint_epochs
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.train.TrainConfig.checkpoint_epochs
```

````

````{py:attribute} shrink_size
:canonical: src.configs.train.TrainConfig.shrink_size
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.train.TrainConfig.shrink_size
```

````

````{py:attribute} post_processing_epochs
:canonical: src.configs.train.TrainConfig.post_processing_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.train.TrainConfig.post_processing_epochs
```

````

````{py:attribute} lr_post_processing
:canonical: src.configs.train.TrainConfig.lr_post_processing
:type: float
:value: >
   0.001

```{autodoc2-docstring} src.configs.train.TrainConfig.lr_post_processing
```

````

````{py:attribute} efficiency_weight
:canonical: src.configs.train.TrainConfig.efficiency_weight
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.train.TrainConfig.efficiency_weight
```

````

````{py:attribute} overflow_weight
:canonical: src.configs.train.TrainConfig.overflow_weight
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.train.TrainConfig.overflow_weight
```

````

````{py:attribute} log_step
:canonical: src.configs.train.TrainConfig.log_step
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.train.TrainConfig.log_step
```

````

````{py:attribute} epoch_start
:canonical: src.configs.train.TrainConfig.epoch_start
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.train.TrainConfig.epoch_start
```

````

````{py:attribute} eval_only
:canonical: src.configs.train.TrainConfig.eval_only
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.train.TrainConfig.eval_only
```

````

````{py:attribute} checkpoint_encoder
:canonical: src.configs.train.TrainConfig.checkpoint_encoder
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.train.TrainConfig.checkpoint_encoder
```

````

````{py:attribute} load_path
:canonical: src.configs.train.TrainConfig.load_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.train.TrainConfig.load_path
```

````

````{py:attribute} resume
:canonical: src.configs.train.TrainConfig.resume
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.train.TrainConfig.resume
```

````

````{py:attribute} logs_dir
:canonical: src.configs.train.TrainConfig.logs_dir
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.train.TrainConfig.logs_dir
```

````

````{py:attribute} model_weights_path
:canonical: src.configs.train.TrainConfig.model_weights_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.train.TrainConfig.model_weights_path
```

````

````{py:attribute} final_model_path
:canonical: src.configs.train.TrainConfig.final_model_path
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.train.TrainConfig.final_model_path
```

````

````{py:attribute} eval_batch_size
:canonical: src.configs.train.TrainConfig.eval_batch_size
:type: int
:value: >
   256

```{autodoc2-docstring} src.configs.train.TrainConfig.eval_batch_size
```

````

````{py:attribute} persistent_workers
:canonical: src.configs.train.TrainConfig.persistent_workers
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.train.TrainConfig.persistent_workers
```

````

````{py:attribute} pin_memory
:canonical: src.configs.train.TrainConfig.pin_memory
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.train.TrainConfig.pin_memory
```

````

`````
