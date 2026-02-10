# {py:mod}`src.pipeline.features.train.model_factory.imitation`

```{py:module} src.pipeline.features.train.model_factory.imitation
```

```{autodoc2-docstring} src.pipeline.features.train.model_factory.imitation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_create_imitation <src.pipeline.features.train.model_factory.imitation._create_imitation>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.imitation._create_imitation
    :summary:
    ```
* - {py:obj}`_create_adaptive_imitation <src.pipeline.features.train.model_factory.imitation._create_adaptive_imitation>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.imitation._create_adaptive_imitation
    :summary:
    ```
* - {py:obj}`_create_critic_helper <src.pipeline.features.train.model_factory.imitation._create_critic_helper>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.imitation._create_critic_helper
    :summary:
    ```
````

### API

````{py:function} _create_imitation(cfg: logic.src.configs.Config, policy, env, kw: typing.Dict[str, typing.Any]) -> pytorch_lightning.LightningModule
:canonical: src.pipeline.features.train.model_factory.imitation._create_imitation

```{autodoc2-docstring} src.pipeline.features.train.model_factory.imitation._create_imitation
```
````

````{py:function} _create_adaptive_imitation(cfg: logic.src.configs.Config, policy, env, kw: typing.Dict[str, typing.Any]) -> pytorch_lightning.LightningModule
:canonical: src.pipeline.features.train.model_factory.imitation._create_adaptive_imitation

```{autodoc2-docstring} src.pipeline.features.train.model_factory.imitation._create_adaptive_imitation
```
````

````{py:function} _create_critic_helper(policy, cfg: logic.src.configs.Config) -> typing.Any
:canonical: src.pipeline.features.train.model_factory.imitation._create_critic_helper

```{autodoc2-docstring} src.pipeline.features.train.model_factory.imitation._create_critic_helper
```
````
