# {py:mod}`src.configs`

```{py:module} src.configs
```

```{autodoc2-docstring} src.configs
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

src.configs.rl
src.configs.policies
src.configs.tasks
src.configs.envs
src.configs.models
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.configs.tracking
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
:type: src.configs.envs.EnvConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.env
```

````

````{py:attribute} model
:canonical: src.configs.Config.model
:type: src.configs.models.ModelConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.model
```

````

````{py:attribute} train
:canonical: src.configs.Config.train
:type: src.configs.tasks.TrainConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.train
```

````

````{py:attribute} optim
:canonical: src.configs.Config.optim
:type: src.configs.models.OptimConfig
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
:type: src.configs.tasks.MetaRLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.meta_rl
```

````

````{py:attribute} hpo
:canonical: src.configs.Config.hpo
:type: src.configs.tasks.HPOConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.hpo
```

````

````{py:attribute} eval
:canonical: src.configs.Config.eval
:type: src.configs.tasks.EvalConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.eval
```

````

````{py:attribute} sim
:canonical: src.configs.Config.sim
:type: src.configs.tasks.SimConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.sim
```

````

````{py:attribute} data
:canonical: src.configs.Config.data
:type: src.configs.tasks.DataConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.data
```

````

````{py:attribute} tracking
:canonical: src.configs.Config.tracking
:type: src.configs.tracking.TrackingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.tracking
```

````

````{py:attribute} must_go
:canonical: src.configs.Config.must_go
:type: src.configs.policies.other.MustGoConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.Config.post_processing
:type: src.configs.policies.other.PostProcessingConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.post_processing
```

````

````{py:attribute} load_dataset
:canonical: src.configs.Config.load_dataset
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.Config.load_dataset
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

````{py:attribute} task
:canonical: src.configs.Config.task
:type: str
:value: >
   'train'

```{autodoc2-docstring} src.configs.Config.task
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

````{py:attribute} run_name
:canonical: src.configs.Config.run_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} src.configs.Config.run_name
```

````

````{py:attribute} start
:canonical: src.configs.Config.start
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.Config.start
```

````

````{py:attribute} p
:canonical: src.configs.Config.p
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.p
```

````

````{py:attribute} callbacks
:canonical: src.configs.Config.callbacks
:type: typing.Dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.Config.callbacks
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
