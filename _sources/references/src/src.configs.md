# {py:mod}`src.configs`

```{py:module} src.configs
```

```{autodoc2-docstring} src.configs
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.configs.rl
src.configs.decoding
src.configs.train
src.configs.env
src.configs.optim
src.configs.hpo
src.configs.model
src.configs.meta_rl
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Config <src.configs.Config>`
  - ```{autodoc2-docstring} src.configs.Config
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.configs.__all__>`
  - ```{autodoc2-docstring} src.configs.__all__
    :summary:
    ```
````

### API

`````{py:class} Config
:canonical: src.configs.Config

```{autodoc2-docstring} src.configs.Config
```

````{py:attribute} env
:canonical: src.configs.Config.env
:type: src.configs.env.EnvConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.env
```

````

````{py:attribute} model
:canonical: src.configs.Config.model
:type: src.configs.model.ModelConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.model
```

````

````{py:attribute} train
:canonical: src.configs.Config.train
:type: src.configs.train.TrainConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.train
```

````

````{py:attribute} optim
:canonical: src.configs.Config.optim
:type: src.configs.optim.OptimConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.optim
```

````

````{py:attribute} rl
:canonical: src.configs.Config.rl
:type: src.configs.rl.RLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.rl
```

````

````{py:attribute} meta_rl
:canonical: src.configs.Config.meta_rl
:type: src.configs.meta_rl.MetaRLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.meta_rl
```

````

````{py:attribute} hpo
:canonical: src.configs.Config.hpo
:type: src.configs.hpo.HPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.hpo
```

````

````{py:attribute} seed
:canonical: src.configs.Config.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.Config.seed
```

````

````{py:attribute} device
:canonical: src.configs.Config.device
:type: str
:value: >
   'cuda'

```{autodoc2-docstring} src.configs.Config.device
```

````

````{py:attribute} experiment_name
:canonical: src.configs.Config.experiment_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.Config.experiment_name
```

````

````{py:attribute} wandb_mode
:canonical: src.configs.Config.wandb_mode
:type: str
:value: >
   'offline'

```{autodoc2-docstring} src.configs.Config.wandb_mode
```

````

````{py:attribute} no_tensorboard
:canonical: src.configs.Config.no_tensorboard
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.Config.no_tensorboard
```

````

````{py:attribute} no_progress_bar
:canonical: src.configs.Config.no_progress_bar
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.Config.no_progress_bar
```

````

````{py:attribute} output_dir
:canonical: src.configs.Config.output_dir
:type: str
:value: >
   'assets/model_weights'

```{autodoc2-docstring} src.configs.Config.output_dir
```

````

````{py:attribute} log_dir
:canonical: src.configs.Config.log_dir
:type: str
:value: >
   'logs'

```{autodoc2-docstring} src.configs.Config.log_dir
```

````

````{py:attribute} run_name
:canonical: src.configs.Config.run_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.Config.run_name
```

````

`````

````{py:data} __all__
:canonical: src.configs.__all__
:value: >
   ['EnvConfig', 'ModelConfig', 'TrainConfig', 'OptimConfig', 'RLConfig', 'MetaRLConfig', 'HPOConfig', ...

```{autodoc2-docstring} src.configs.__all__
```

````
