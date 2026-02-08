# {py:mod}`src.pipeline.features.train.model_factory.ppo`

```{py:module} src.pipeline.features.train.model_factory.ppo
```

```{autodoc2-docstring} src.pipeline.features.train.model_factory.ppo
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_create_ppo_family <src.pipeline.features.train.model_factory.ppo._create_ppo_family>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.ppo._create_ppo_family
    :summary:
    ```
* - {py:obj}`_create_gdpo <src.pipeline.features.train.model_factory.ppo._create_gdpo>`
  - ```{autodoc2-docstring} src.pipeline.features.train.model_factory.ppo._create_gdpo
    :summary:
    ```
````

### API

````{py:function} _create_ppo_family(algo_name: str, cfg: logic.src.configs.Config, policy, env, kw: typing.Dict[str, typing.Any]) -> pytorch_lightning.LightningModule
:canonical: src.pipeline.features.train.model_factory.ppo._create_ppo_family

```{autodoc2-docstring} src.pipeline.features.train.model_factory.ppo._create_ppo_family
```
````

````{py:function} _create_gdpo(cfg: logic.src.configs.Config, policy, env, kw: typing.Dict[str, typing.Any]) -> pytorch_lightning.LightningModule
:canonical: src.pipeline.features.train.model_factory.ppo._create_gdpo

```{autodoc2-docstring} src.pipeline.features.train.model_factory.ppo._create_gdpo
```
````
