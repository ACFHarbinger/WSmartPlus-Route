# {py:mod}`src.pipeline.features.train.model_factory.builder`

```{py:module} src.pipeline.features.train.model_factory.builder
```

```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_model <src.pipeline.features.train.model_factory.builder.create_model>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder.create_model
    :summary:
    ```
* - {py:obj}`_init_environment <src.pipeline.features.train.model_factory.builder._init_environment>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._init_environment
    :summary:
    ```
* - {py:obj}`_init_policy <src.pipeline.features.train.model_factory.builder._init_policy>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._init_policy
    :summary:
    ```
* - {py:obj}`_init_hybrid_policy <src.pipeline.features.train.model_factory.builder._init_hybrid_policy>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._init_hybrid_policy
    :summary:
    ```
* - {py:obj}`_prepare_rl_kwargs <src.pipeline.features.train.model_factory.builder._prepare_rl_kwargs>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._prepare_rl_kwargs
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.pipeline.features.train.model_factory.builder.logger>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.pipeline.features.train.model_factory.builder.logger
:value: >
   'get_pylogger(...)'

```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder.logger
```

````

````{py:function} create_model(cfg: logic.src.configs.Config) -> pytorch_lightning.LightningModule
:canonical: src.pipeline.features.train.model_factory.builder.create_model

```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder.create_model
```
````

````{py:function} _init_environment(cfg: logic.src.configs.Config)
:canonical: src.pipeline.features.train.model_factory.builder._init_environment

```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._init_environment
```
````

````{py:function} _init_policy(cfg: logic.src.configs.Config, env: typing.Any)
:canonical: src.pipeline.features.train.model_factory.builder._init_policy

```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._init_policy
```
````

````{py:function} _init_hybrid_policy(cfg: logic.src.configs.Config)
:canonical: src.pipeline.features.train.model_factory.builder._init_hybrid_policy

```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._init_hybrid_policy
```
````

````{py:function} _prepare_rl_kwargs(cfg: logic.src.configs.Config, env: typing.Any, policy: typing.Any)
:canonical: src.pipeline.features.train.model_factory.builder._prepare_rl_kwargs

```{autodoc2-docstring} src.pipeline.features.train.model_factory.builder._prepare_rl_kwargs
```
````
